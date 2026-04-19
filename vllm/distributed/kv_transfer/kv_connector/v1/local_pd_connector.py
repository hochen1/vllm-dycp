"""
LocalPDConnector: File-based KV connector for local Prefill-Decode separation.

In local PD separation mode:
- Prefill phase: each CP rank independently saves its KV slice to files
- Decode phase: the single decode rank loads all KV slices and reconstructs
  the full KV cache

File layout:
  {storage_path}/{pd_request_prefix}/
    meta.json              - metadata (cp_world_size, token ranges, etc.)
    rank_{i}/layer_{j}.safetensors - per-rank, per-layer KV data

KNOWN LIMITATION (v1):
  KV reconstruction uses simple concatenation (torch.cat) which assumes
  ranks hold contiguous token ranges. With DualChunkSwap, tokens are
  interleaved across ranks in a head/tail pattern, so concatenation may
  produce incorrectly ordered KV. This needs to be addressed by saving
  the PCPManager's restore index (pcp_allgather_restore_idx) and applying
  it during load. For initial testing, validate KV correctness by comparing
  outputs with and without PD separation.
"""

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import safetensors.torch
import torch

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    align_to_block_size,
)
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import MLACommonMetadata
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Valid prefix pattern: alphanumeric + hyphens only
_VALID_PREFIX_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


# ---------------------------------------------------------------------------
# Metadata structures
# ---------------------------------------------------------------------------

@dataclass
class LocalPDReqMeta:
    """Per-request metadata for LocalPD connector."""
    req_id: str
    token_ids: torch.Tensor
    slot_mapping: torch.Tensor
    is_store: bool
    pd_request_prefix: str

    @staticmethod
    def make_meta(
        req_id: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
        pd_request_prefix: str,
        num_tokens_override: int | None = None,
    ) -> "LocalPDReqMeta":
        # num_tokens_override allows specifying exact token count
        # (e.g., tokens_per_rank from DualChunkSwap) instead of
        # deriving from block count which may include unused slots.
        if num_tokens_override is not None:
            valid_num_tokens = num_tokens_override
        else:
            valid_num_tokens = len(block_ids) * block_size
            valid_num_tokens = min(valid_num_tokens, len(token_ids))
        token_ids_tensor = torch.tensor(token_ids)[:valid_num_tokens]
        block_ids_tensor = torch.tensor(block_ids)
        num_blocks = block_ids_tensor.shape[0]
        block_offsets = torch.arange(0, block_size)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids_tensor.reshape((num_blocks, 1)) * block_size
        )
        slot_mapping = slot_mapping.flatten()[:valid_num_tokens]
        return LocalPDReqMeta(
            req_id=req_id,
            token_ids=token_ids_tensor,
            slot_mapping=slot_mapping,
            is_store=is_store,
            pd_request_prefix=pd_request_prefix,
        )


@dataclass
class LocalPDConnectorMetadata(KVConnectorMetadata):
    """Connector metadata passed from scheduler to worker."""
    requests: list[LocalPDReqMeta] = field(default_factory=list)
    cp_rank: int = 0

    def add_request(
        self,
        req_id: str,
        token_ids: list[int],
        block_ids: list[int],
        block_size: int,
        is_store: bool,
        pd_request_prefix: str,
        num_tokens_override: int | None = None,
    ) -> None:
        self.requests.append(
            LocalPDReqMeta.make_meta(
                req_id=req_id,
                token_ids=token_ids,
                block_ids=block_ids,
                block_size=block_size,
                is_store=is_store,
                pd_request_prefix=pd_request_prefix,
                num_tokens_override=num_tokens_override,
            )
        )


# ---------------------------------------------------------------------------
# Helper: KV extract / inject
# ---------------------------------------------------------------------------

def _extract_kv_from_layer(
    kv_layer: torch.Tensor,
    slot_mapping: torch.Tensor,
    is_mla: bool,
) -> torch.Tensor:
    """Extract KV cache from paged buffer using slot_mapping."""
    if is_mla:
        num_pages, page_size = kv_layer.shape[0], kv_layer.shape[1]
        return kv_layer.reshape(num_pages * page_size, -1)[slot_mapping, ...]
    num_pages, page_size = kv_layer.shape[1], kv_layer.shape[2]
    return kv_layer.reshape(2, num_pages * page_size, -1)[:, slot_mapping, ...]


def _inject_kv_into_layer(
    dst_kv_layer: torch.Tensor,
    src_kv: torch.Tensor,
    slot_mapping: torch.Tensor,
    is_mla: bool,
) -> None:
    """Inject KV cache into paged buffer using slot_mapping."""
    dst_shape = dst_kv_layer.shape
    if is_mla:
        num_pages, page_size = dst_shape[0], dst_shape[1]
        dst_flat = dst_kv_layer.reshape(num_pages * page_size, -1)
        dst_flat[slot_mapping, ...] = src_kv
    else:
        num_pages, page_size = dst_shape[1], dst_shape[2]
        dst_flat = dst_kv_layer.reshape(2, num_pages * page_size, -1)
        dst_flat[:, slot_mapping, ...] = src_kv


def _compute_dualchunkswap_restore_idx(
    num_tokens: int,
    cp_world_size: int,
) -> tuple[torch.Tensor, int]:
    """Compute the restore index for DualChunkSwap KV reordering.

    After CP prefill, each rank holds KV for tokens assigned by the
    DualChunkSwap pattern (head/tail interleaving). When we concatenate
    all ranks' KV, the token order follows this pattern. This function
    computes the index needed to restore original token order.

    Mirrors the logic in PCPManager.update_tokens_for_pcp() lines 260-270.

    Args:
        num_tokens: Number of actual (unpadded) prompt tokens.
        cp_world_size: Number of CP ranks.

    Returns:
        (restore_idx, padded_n): restore_idx[i] gives the index in
        the concatenated KV tensor that holds KV for original position i.
        padded_n is the total padded token count.
    """
    import numpy as np

    W = cp_world_size
    padded_n = int(np.ceil(num_tokens / (2 * W)) * (2 * W))
    tokens_per_rank = padded_n // W
    chunk_size = tokens_per_rank // 2

    # Build positions for each rank following DualChunkSwap
    all_positions = []
    for rank in range(W):
        head_start = rank * chunk_size
        tail_start = (2 * W - 1 - rank) * chunk_size
        # Head chunk positions
        for i in range(chunk_size):
            all_positions.append(head_start + i)
        # Tail chunk positions
        for i in range(chunk_size):
            all_positions.append(tail_start + i)

    all_positions_np = np.array(all_positions, dtype=np.int64)
    restore_idx = all_positions_np.argsort()
    return torch.from_numpy(restore_idx.copy()), padded_n


def _validate_prefix(prefix: str) -> bool:
    """Validate prefix to prevent path traversal."""
    return bool(prefix and _VALID_PREFIX_RE.match(prefix))


def _atomic_write_json(path: str, data: dict) -> None:
    """Write JSON atomically using rename to prevent partial reads."""
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, path)  # atomic on POSIX
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# LocalPDConnector
# ---------------------------------------------------------------------------

class LocalPDConnector(KVConnectorBase_V1):
    """
    File-based KV connector for local Prefill-Decode separation.

    Scheduler side:
    - Detects prefill vs decode requests via kv_transfer_params
    - For prefill: marks KV for saving (is_store=True)
    - For decode: checks KV availability, marks for loading (is_store=False)

    Worker side:
    - Prefill: each CP rank saves its KV slice to files
    - Decode: single rank loads all slices and reconstructs full KV
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._block_size = vllm_config.cache_config.block_size
        self._storage_path = self._kv_transfer_config.get_from_extra_config(
            "shared_storage_path", "/tmp/kv_cache"
        )
        self._cp_world_size = vllm_config.parallel_config.dp_per_domain

        if role == KVConnectorRole.SCHEDULER:
            self._cross_requests_need_load: list[dict[str, "Request"]] = [
                {} for _ in range(self._cp_world_size)
            ]
            self._prefill_requests: dict[str, dict[str, Any]] = {}
            # In-memory flags for GPU buffer mode (no file I/O)
            self._completed_prefills: dict[str, dict[str, Any]] = {}

        if role == KVConnectorRole.WORKER:
            # GPU memory buffer: {prefix: {layer_name: tensor}}
            # Keeps all-gathered KV on GPU between prefill and decode.
            self._gpu_kv_buffer: dict[str, dict[str, torch.Tensor]] = {}
            # Pending local KV to be batch-all-gathered in wait_for_save
            self._pending_local_kv: dict[str, list[tuple[str, torch.Tensor]]] = {}
            self._pending_cp_rank: int = 0

        logger.info(
            "LocalPDConnector initialized: storage_path=%s, "
            "cp_world_size=%d, block_size=%d",
            self._storage_path, self._cp_world_size, self._block_size,
        )

    # ==============================
    # Scheduler-side methods
    # ==============================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        kv_params = request.kv_transfer_params
        logger.info(
            "get_num_new_matched_tokens called: req=%s, "
            "num_computed=%d, kv_params=%s",
            request.request_id,
            num_computed_tokens,
            kv_params,
        )
        if not kv_params:
            return 0, False

        if kv_params.get("do_remote_prefill"):
            # Decode request: check if prefill KV files are ready
            prefix = kv_params.get("pd_request_prefix")
            if not prefix or not _validate_prefix(prefix):
                logger.warning(
                    "Decode request has invalid pd_request_prefix: %s",
                    prefix,
                )
                return 0, False

            # Check in-memory flag first (GPU buffer mode)
            meta = self._completed_prefills.get(prefix)
            if meta is None:
                # Fallback: check file (for backward compatibility)
                meta_path = os.path.join(
                    self._storage_path, prefix, "meta.json"
                )
                if not os.path.exists(meta_path):
                    logger.debug(
                        "KV not ready for prefix=%s, will retry", prefix
                    )
                    return None, False
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(
                        "Failed to read meta for prefix=%s: %s",
                        prefix, e,
                    )
                    return None, False

            if not meta.get("completed"):
                return None, False

            num_prompt_tokens = meta["num_prompt_tokens"]
            # For PD separation, we want to load all prompt KV even
            # if it doesn't fill a complete block. Use block-aligned
            # token count but ensure at least 1 token is loaded.
            aligned = align_to_block_size(num_prompt_tokens, self._block_size)
            if aligned == 0 and num_prompt_tokens > 0:
                # Prompt shorter than block_size: still load what we have.
                # The scheduler will allocate 1 block for these tokens.
                aligned = num_prompt_tokens - 1  # -1 because last token needs compute
            ext_tokens = aligned - num_computed_tokens
            if ext_tokens <= 0:
                return 0, False

            logger.info(
                "External KV found for prefix=%s: %d tokens (aligned=%d)",
                prefix, num_prompt_tokens, aligned,
            )
            # Use synchronous mode (False): the scheduler treats these
            # tokens as already computed. The actual KV file loading
            # happens in start_load_kv() during the forward pass.
            # Async mode (True) would require WAITING_FOR_REMOTE_KVS
            # state which needs a separate forward pass to trigger
            # the connector's get_finished() — causing a deadlock
            # in single-instance PD separation.
            return ext_tokens, False

        if kv_params.get("do_remote_decode"):
            # Prefill request: track for KV saving, execute normally
            self._prefill_requests[request.request_id] = kv_params
            logger.info(
                "Tracked prefill request %s for KV saving (prefix=%s)",
                request.request_id,
                kv_params.get("pd_request_prefix"),
            )
            return 0, False

        logger.info(
            "Request %s has kv_transfer_params but no PD flags: %s",
            request.request_id, kv_params,
        )
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ):
        if num_external_tokens > 0:
            for cp_rank in request.cp_ranks:
                self._cross_requests_need_load[cp_rank][
                    request.request_id
                ] = request

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = LocalPDConnectorMetadata(cp_rank=scheduler_output.cp_rank)
        total_need_load = 0
        cp_rank = scheduler_output.cp_rank

        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []

            if new_req.req_id in self._cross_requests_need_load[cp_rank]:
                # Decode request: load KV from files
                request = self._cross_requests_need_load[cp_rank][
                    new_req.req_id
                ]
                pd_prefix = (
                    request.kv_transfer_params.get("pd_request_prefix", "")
                    if request.kv_transfer_params
                    else ""
                )
                meta.add_request(
                    req_id=new_req.req_id,
                    token_ids=token_ids,
                    block_ids=new_req.block_ids[0],
                    block_size=self._block_size,
                    is_store=False,
                    pd_request_prefix=pd_prefix,
                )
                total_need_load += 1
            else:
                # Prefill request: save KV to files.
                # Use get() not pop() because build_connector_meta is
                # called once per CP rank — all ranks need to see the
                # same prefill request. Clean up after last rank.
                kv_params = self._prefill_requests.get(
                    new_req.req_id, None
                )
                if kv_params and kv_params.get("do_remote_decode"):
                    pd_prefix = kv_params.get("pd_request_prefix", "")
                    # Determine actual CP size for this request
                    cp_size = scheduler_output.cp_rank_scheduled_tokens.get(
                        new_req.req_id, 1
                    )
                    num_scheduled = scheduler_output.num_scheduled_tokens.get(
                        new_req.req_id, len(token_ids)
                    )
                    if cp_size > 1:
                        # Full CP: compute tokens_per_rank from
                        # DualChunkSwap alignment.
                        import numpy as np
                        padded = int(
                            np.ceil(num_scheduled / (2 * cp_size))
                            * (2 * cp_size)
                        )
                        tpr = padded // cp_size
                    else:
                        # Single rank: save all scheduled tokens
                        tpr = None  # use default logic
                    meta.add_request(
                        req_id=new_req.req_id,
                        token_ids=token_ids,
                        block_ids=new_req.block_ids[0],
                        block_size=self._block_size,
                        is_store=True,
                        pd_request_prefix=pd_prefix,
                        num_tokens_override=tpr,
                    )

        # Clean up prefill tracking after last CP rank processes it
        if cp_rank == self._cp_world_size - 1:
            for new_req in scheduler_output.scheduled_new_reqs:
                self._prefill_requests.pop(new_req.req_id, None)

        # Handle cached/resumed requests needing KV load
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            resumed = req_id in cached_reqs.resumed_req_ids
            if (
                not resumed
                or req_id not in self._cross_requests_need_load[cp_rank]
            ):
                continue

            request = self._cross_requests_need_load[cp_rank][req_id]
            pd_prefix = (
                request.kv_transfer_params.get("pd_request_prefix", "")
                if request.kv_transfer_params
                else ""
            )
            num_computed_tokens = cached_reqs.num_computed_tokens[i]
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            new_block_ids = cached_reqs.new_block_ids[i]

            total_tokens = num_computed_tokens + num_new_tokens
            token_ids = request.all_token_ids[:total_tokens]

            assert new_block_ids is not None
            block_ids = new_block_ids[0]

            meta.add_request(
                req_id=req_id,
                token_ids=token_ids,
                block_ids=block_ids,
                block_size=self._block_size,
                is_store=False,
                pd_request_prefix=pd_prefix,
            )
            total_need_load += 1

        store_count = sum(1 for r in meta.requests if r.is_store)
        load_count = sum(1 for r in meta.requests if not r.is_store)
        logger.info(
            "build_connector_meta cp_rank=%d: %d store, %d load requests",
            cp_rank, store_count, load_count,
        )

        expected = len(self._cross_requests_need_load[cp_rank])
        if total_need_load != expected:
            logger.warning(
                "LocalPDConnector: need_load mismatch on cp_rank=%d: "
                "total_need_load=%d, expected=%d. "
                "Some requests may not have been scheduled this step.",
                cp_rank,
                total_need_load,
                expected,
            )
        self._cross_requests_need_load[cp_rank].clear()
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        kv_params = request.kv_transfer_params
        if not kv_params:
            return False, None

        if kv_params.get("do_remote_decode"):
            # Prefill finished: store metadata in memory for decode
            prefix = kv_params.get("pd_request_prefix", "")
            if not _validate_prefix(prefix):
                logger.error("Invalid prefix in request_finished: %s", prefix)
                return False, None

            actual_cp_count = len(request.cp_ranks) if request.cp_ranks else 1
            meta = {
                "completed": True,
                "cp_world_size": actual_cp_count,
                "num_prompt_tokens": len(request.prompt_token_ids),
                "block_size": self._block_size,
                "pd_request_prefix": prefix,
            }

            # Store in memory (no file I/O)
            self._completed_prefills[prefix] = meta

            logger.info(
                "Prefill finished for prefix=%s "
                "(num_prompt_tokens=%d, cp_world_size=%d)",
                prefix,
                len(request.prompt_token_ids),
                actual_cp_count,
            )

            return_params = {
                "pd_request_prefix": prefix,
                "cp_world_size": actual_cp_count,
                "num_prompt_tokens": len(request.prompt_token_ids),
            }
            return False, return_params

        if kv_params.get("do_remote_prefill"):
            # Decode finished: clean up in-memory metadata
            prefix = kv_params.get("pd_request_prefix", "")
            self._completed_prefills.pop(prefix, None)
            self._prefill_requests.pop(request.request_id, None)
            return False, None

        return False, None

    def _cleanup_kv_files(self, prefix: str) -> None:
        """Remove KV files after decode completes."""
        import shutil

        kv_dir = os.path.join(self._storage_path, prefix)
        if not os.path.exists(kv_dir):
            return
        try:
            shutil.rmtree(kv_dir)
            logger.debug("Cleaned up KV files for prefix=%s", prefix)
        except OSError as e:
            logger.warning(
                "Failed to clean up KV files for prefix=%s: %s", prefix, e
            )

    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_kv(
        self, forward_context: "ForwardContext", **kwargs: Any
    ) -> None:
        """Load KV cache from files for decode requests."""
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, LocalPDConnectorMetadata):
            return

        # Detect MLA by inspecting the KV cache layer shape.
        # MLA: [num_pages, page_size, kv_dim] (3D)
        # Non-MLA: [2, num_pages, page_size, kv_dim] (4D)
        # We determine this from the first attention layer's cache.
        is_mla = None  # Will be detected from layer shape

        for req_meta in metadata.requests:
            if req_meta.is_store:
                continue

            prefix = req_meta.pd_request_prefix

            # Check GPU buffer first (fast path)
            kv_buf = self._gpu_kv_buffer.get(prefix)
            if kv_buf is None:
                logger.error(
                    "No GPU KV buffer for prefix=%s", prefix
                )
                continue

            actual_tokens = req_meta.slot_mapping.shape[0]
            logger.info(
                "Loading KV from GPU buffer for prefix=%s "
                "(%d layers, %d tokens)",
                prefix, len(kv_buf), actual_tokens,
            )

            import time as _time
            _t0 = _time.monotonic()

            sm = req_meta.slot_mapping
            layers_injected = 0
            for layer_name, layer in forward_context.no_compile_layers.items():
                kv_cache_attr = getattr(layer, "kv_cache", None)
                if kv_cache_attr is None:
                    continue
                kv_cache_layer = kv_cache_attr[
                    forward_context.virtual_engine
                ]

                if is_mla is None:
                    is_mla = kv_cache_layer.dim() == 3

                full_kv = kv_buf.get(layer_name)
                if full_kv is None:
                    continue

                # Inject: scatter KV into paged buffer
                if is_mla:
                    num_pages = kv_cache_layer.shape[0]
                    page_size = kv_cache_layer.shape[1]
                    flat = kv_cache_layer.reshape(
                        num_pages * page_size, -1
                    )
                    flat[sm] = full_kv[:actual_tokens]
                else:
                    num_pages = kv_cache_layer.shape[1]
                    page_size = kv_cache_layer.shape[2]
                    flat = kv_cache_layer.reshape(
                        2, num_pages * page_size, -1
                    )
                    flat[:, sm] = full_kv[:, :actual_tokens]
                layers_injected += 1

            # Free GPU buffer
            del self._gpu_kv_buffer[prefix]

            _elapsed = (_time.monotonic() - _t0) * 1000
            logger.info(
                "KV injected for prefix=%s: %d layers in %.1fms",
                prefix, layers_injected, _elapsed,
            )

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        """Save KV cache for prefill requests.

        For multi-rank CP: uses raw pre-mask KV from attention args
        (passed via kwargs by the decorator) to all-gather complete KV.
        For single-rank: extracts directly from paged buffer.
        """
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, LocalPDConnectorMetadata):
            return

        is_mla = isinstance(attn_metadata, MLACommonMetadata)
        cp_rank = metadata.cp_rank
        raw_kv_c_normed = kwargs.get("_raw_kv_c_normed")
        raw_k_pe = kwargs.get("_raw_k_pe")

        for req_meta in metadata.requests:
            if not req_meta.is_store:
                continue

            prefix = req_meta.pd_request_prefix
            num_tokens = req_meta.slot_mapping.shape[0]

            # Determine CP mode from scheduler metadata
            cp_size = getattr(attn_metadata, "num_dycp_reqs", 0)
            is_multi_cp = (
                self._cp_world_size > 1
                and cp_size > 0
                and raw_kv_c_normed is not None
                and is_mla
            )

            if is_multi_cp:
                self._save_kv_allgather_raw(
                    layer_name, raw_kv_c_normed, raw_k_pe,
                    attn_metadata, num_tokens,
                    cp_rank, prefix,
                )
            else:
                # Single CP rank: extract from paged buffer, store in GPU buffer
                sm = req_meta.slot_mapping
                kv_data = _extract_kv_from_layer(kv_layer, sm, is_mla)
                self._gpu_kv_buffer.setdefault(prefix, {})[layer_name] = (
                    kv_data.detach()
                )

    def _save_kv_allgather_raw(
        self,
        layer_name: str,
        raw_kv_c_normed: torch.Tensor,
        raw_k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
        num_tokens: int,
        cp_rank: int,
        prefix: str,
    ) -> None:
        """Buffer pre-mask local KV for batch all-gather in wait_for_save.

        Instead of doing NCCL all-gather per layer (27 times),
        we buffer the local KV and do a single batch all-gather
        in wait_for_save() after all layers complete.
        """
        num_actual = getattr(attn_metadata, "num_actual_tokens", num_tokens)
        local_kv = torch.cat([
            raw_kv_c_normed[:num_actual],
            raw_k_pe[:num_actual].squeeze(1),
        ], dim=-1)

        # Buffer for batch processing
        self._pending_local_kv.setdefault(prefix, []).append(
            (layer_name, local_kv.detach())
        )
        self._pending_cp_rank = cp_rank
        # Store restore index (same for all layers, save once)
        if not hasattr(self, "_pending_restore_idx"):
            self._pending_restore_idx = None
        restore_idx = getattr(
            attn_metadata, "pcp_allgather_restore_idx", None
        )
        if restore_idx is not None:
            self._pending_restore_idx = restore_idx

    def wait_for_save(self):
        """Batch all-gather all pending local KV across CP ranks."""
        if not self._pending_local_kv:
            return

        import time as _time
        _t0 = _time.monotonic()

        from vllm.distributed.parallel_state import get_dycp_group
        dycp_group = get_dycp_group()
        cp_rank = self._pending_cp_rank
        restore_idx = getattr(self, "_pending_restore_idx", None)

        for prefix, layer_kvs in self._pending_local_kv.items():
            # Concatenate all layers' local KV into one tensor
            # for a single all-gather call
            layer_names = [name for name, _ in layer_kvs]
            local_tensors = [kv for _, kv in layer_kvs]
            # All same shape [num_actual, kv_dim]
            stacked = torch.cat(local_tensors, dim=0)  # [num_layers * N, D]

            # Single all-gather for all layers
            gathered = dycp_group.all_gather(
                stacked.contiguous(), dim=0
            )

            # Only rank_0 reconstructs
            if cp_rank == 0:
                tokens_per_layer = local_tensors[0].shape[0]
                num_layers = len(layer_names)
                W = self._cp_world_size

                # Split gathered back into per-rank, per-layer chunks
                # gathered shape: [W * num_layers * N, D]
                # Layout: [rank0_layer0, rank0_layer1, ..., rank0_layerN,
                #          rank1_layer0, ..., rankW_layerN]
                for layer_idx, layer_name in enumerate(layer_names):
                    # Collect this layer's data from each rank
                    layer_chunks = []
                    for r in range(W):
                        start = (r * num_layers + layer_idx) * tokens_per_layer
                        end = start + tokens_per_layer
                        layer_chunks.append(gathered[start:end])

                    layer_gathered = torch.cat(layer_chunks, dim=0)

                    # Apply DualChunkSwap restore
                    if restore_idx is not None and restore_idx.shape[0] > 0:
                        total = layer_gathered.shape[0]
                        ri = restore_idx[:total].to(layer_gathered.device)
                        ri = torch.clamp(ri, 0, total - 1)
                        layer_gathered = layer_gathered[ri]

                    self._gpu_kv_buffer.setdefault(prefix, {})[layer_name] = (
                        layer_gathered
                    )

        self._pending_local_kv.clear()
        self._pending_restore_idx = None

        _elapsed = (_time.monotonic() - _t0) * 1000
        logger.info("wait_for_save: all-gather + restore in %.1fms", _elapsed)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Report decode requests as finished receiving after KV load."""
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, LocalPDConnectorMetadata):
            return None, None

        # All load (decode) requests are synchronously loaded in
        # start_load_kv, so report them all as finished receiving.
        load_req_ids = {
            r.req_id for r in metadata.requests if not r.is_store
        }

        return None, load_req_ids if load_req_ids else None
