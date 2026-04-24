"""
LocalPDConnector: KV connector for local Prefill-Decode separation.

In local PD separation mode:
- Prefill phase: all CP ranks save KV to shared storage (IPC or GPU buffer)
- Decode phase: the single decode rank loads KV and injects into paged cache

Two KV transfer modes:
1. CUDA IPC (default for multi-CP): direct GPU-to-GPU memcpy via IPC handles
2. GPU buffer (fallback for single-CP): all-gather + in-memory buffer

KNOWN LIMITATION (v1):
  KV reconstruction uses simple concatenation (torch.cat) which assumes
  ranks hold contiguous token ranges. With DualChunkSwap, tokens are
  interleaved across ranks in a head/tail pattern, so concatenation may
  produce incorrectly ordered KV. This needs to be addressed by saving
  the PCPManager's restore index (pcp_allgather_restore_idx) and applying
  it during load. For initial testing, validate KV correctness by comparing
  outputs with and without PD separation.
"""

import ctypes
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
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
    has_store_requests: bool = False
    # IPC metadata: {prefix: {per_rank_block_ids, interleave_size, ...}}
    ipc_prefill_metas: dict[str, dict[str, Any]] = field(default_factory=dict)

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
        if is_store:
            self.has_store_requests = True


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
        self._interleave_size = vllm_config.parallel_config.cp_kv_cache_interleave_size

        if role == KVConnectorRole.SCHEDULER:
            self._cross_requests_need_load: list[dict[str, "Request"]] = [
                {} for _ in range(self._cp_world_size)
            ]
            self._prefill_requests: dict[str, dict[str, Any]] = {}
            # In-memory flags for GPU buffer mode (no file I/O)
            self._completed_prefills: dict[str, dict[str, Any]] = {}
            # IPC: {prefix: prefill_request_id} for delayed block freeing
            self._ipc_delayed_prefill_ids: dict[str, str] = {}

        if role == KVConnectorRole.WORKER:
            self._gpu_kv_buffer: dict[str, dict[str, torch.Tensor]] = {}
            self._pending_local_kv: dict[str, list[tuple[str, torch.Tensor]]] = {}
            # IPC state (populated in register_kv_caches)
            self._cuda_lib = None
            # {src_rank: {layer_name: (remote_ptr, shape, dtype, elem_size)}}
            self._remote_ipc_info: dict[int, dict[str, tuple]] = {}
            self._local_kv_caches: dict[str, torch.Tensor] = {}
            self._ipc_initialized: bool = False
            self._ipc_stream = None  # dedicated CUDA stream for async IPC
            # {decode_req_id: (cuda_event, prefill_req_id)} async IPC tracking
            self._ipc_pending_events: dict[str, tuple] = {}

        logger.info(
            "LocalPDConnector initialized: storage_path=%s, "
            "cp_world_size=%d, block_size=%d",
            self._storage_path, self._cp_world_size, self._block_size,
        )

    # ==============================
    # IPC Handle Exchange
    # ==============================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Exchange IPC handles for paged buffer tensors across CP ranks.

        Called once during worker initialization after paged buffers are
        allocated. Each rank gets IPC pointers to all other ranks' paged
        buffers, enabling single-sided reads during decode.
        """
        self._local_kv_caches = kv_caches

        if self._cp_world_size <= 1:
            logger.info("IPC skipped: cp_world_size=%d", self._cp_world_size)
            return

        if os.environ.get("VLLM_DYCP_USE_IPC", "1") == "0":
            logger.info("IPC disabled via VLLM_DYCP_USE_IPC=0")
            return

        import time as _time
        t0 = _time.monotonic()

        from vllm.distributed.device_communicators.cuda_wrapper import (
            CudaRTLibrary,
            cudaIpcMemHandle_t,
        )
        from vllm.distributed.parallel_state import get_dycp_group

        dycp_group = get_dycp_group()
        my_rank = dycp_group.rank_in_group
        world_size = dycp_group.world_size

        self._cuda_lib = CudaRTLibrary()

        # For each layer, get local IPC handle and exchange with all ranks
        layer_handles: dict[str, list[bytes]] = {}

        for layer_name, kv_tensor in kv_caches.items():
            handle = self._cuda_lib.cudaIpcGetMemHandle(
                ctypes.c_void_p(kv_tensor.data_ptr())
            )
            handle_bytes = bytes(handle)

            # All-to-all handle exchange via sequential broadcasts
            all_handles: list[bytes | None] = [None] * world_size
            all_handles[my_rank] = handle_bytes
            for src in range(world_size):
                obj_list = [all_handles[src]]
                dycp_group.broadcast_object_list(obj_list, src=src)
                all_handles[src] = obj_list[0]

            layer_handles[layer_name] = all_handles  # type: ignore[assignment]

        # Open remote handles and store raw pointers + metadata
        for layer_name, kv_tensor in kv_caches.items():
            elem_size = kv_tensor.element_size()
            for src_rank in range(world_size):
                if src_rank == my_rank:
                    continue

                handle_bytes = layer_handles[layer_name][src_rank]
                handle = cudaIpcMemHandle_t()
                ctypes.memmove(
                    ctypes.byref(handle), handle_bytes, 128
                )
                remote_ptr = self._cuda_lib.cudaIpcOpenMemHandle(handle)

                # Store raw pointer + metadata for cudaMemcpy at load time
                self._remote_ipc_info.setdefault(
                    src_rank, {}
                )[layer_name] = (
                    remote_ptr, kv_tensor.shape, kv_tensor.dtype, elem_size,
                )

        # Create dedicated CUDA stream + event for async IPC transfers
        stream_ptr = ctypes.c_void_p()
        self._cuda_lib.CUDART_CHECK(
            self._cuda_lib.funcs["cudaStreamCreate"](
                ctypes.byref(stream_ptr)
            )
        )
        self._ipc_stream = stream_ptr

        event_ptr = ctypes.c_void_p()
        self._cuda_lib.CUDART_CHECK(
            self._cuda_lib.funcs["cudaEventCreate"](
                ctypes.byref(event_ptr)
            )
        )
        self._ipc_event = event_ptr

        self._ipc_initialized = True
        elapsed = (_time.monotonic() - t0) * 1000
        logger.info(
            "IPC handles exchanged: %d layers x %d ranks in %.1fms "
            "(async stream created)",
            len(kv_caches), world_size, elapsed,
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
                # Reconstruct from decode request's kv_transfer_params
                # (proxy-mediated flow: metadata arrives with the request)
                if kv_params.get("per_rank_block_ids"):
                    meta = {
                        "completed": True,
                        "per_rank_block_ids": kv_params["per_rank_block_ids"],
                        "num_prompt_tokens": kv_params.get("num_prompt_tokens", 0),
                        "cp_world_size": kv_params.get("cp_world_size", 1),
                        "interleave_size": kv_params.get("interleave_size", self._interleave_size),
                        "block_size": self._block_size,
                        "pd_request_prefix": prefix,
                    }
                    self._completed_prefills[prefix] = meta
                    logger.info(
                        "Reconstructed prefill meta from kv_params "
                        "for prefix=%s", prefix,
                    )
            if meta is None:
                logger.debug(
                    "KV not ready for prefix=%s, will retry", prefix
                )
                return None, False

            if not meta.get("completed"):
                return None, False

            num_prompt_tokens = meta["num_prompt_tokens"]
            # In PD separation, the prefill instance computes KV cache
            # for ALL prompt tokens including the last block. Use
            # num_prompt_tokens - 1 (not align_to_block_size) because:
            # 1. align_to_block_size drops up to (block_size-1) tokens,
            #    causing an extra prefill step for the remainder.
            # 2. The last prompt token's KV is already computed by
            #    prefill, so only 1 new token (first decode) is needed.
            # The scheduler handles non-block-aligned num_computed_tokens
            # correctly — block allocation uses num_new_tokens +
            # num_external_computed_tokens which is still block-aligned.
            aligned = num_prompt_tokens - 1
            if aligned < 0:
                aligned = 0
            ext_tokens = aligned - num_computed_tokens
            if ext_tokens <= 0:
                return 0, False

            import time as _time
            _now = _time.monotonic() * 1000
            _gap = _now - meta.get("_finish_time_ms", _now)
            logger.info(
                "External KV found for prefix=%s: %d tokens "
                "(aligned=%d, gap_from_prefill=%.1fms)",
                prefix, num_prompt_tokens, aligned, _gap,
            )
            # Sync mode: request is scheduled normally. IPC copy runs
            # on dedicated stream, GPU-level dependency via
            # cudaStreamWaitEvent in wait_for_layer_load ensures KV
            # is ready before attention. No CPU blocking.
            return ext_tokens, False

        if kv_params.get("do_remote_decode"):
            # Prefill request: track for KV saving, execute normally
            self._prefill_requests[request.request_id] = kv_params
            logger.info(
                "Tracked prefill req=%s, _prefill_requests now has %d entries",
                request.request_id, len(self._prefill_requests),
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

        if cp_rank == 0:
            logger.info(
                "build_connector_meta START: _prefill_requests=%s, "
                "new_reqs=%d, num_sched=%s",
                list(self._prefill_requests.keys())[:3],
                len(scheduler_output.scheduled_new_reqs),
                list(scheduler_output.num_scheduled_tokens.keys())[:3],
            )

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
                # Attach IPC metadata for worker-side KV loading
                prefill_meta = self._completed_prefills.get(pd_prefix)
                if prefill_meta:
                    meta.ipc_prefill_metas[pd_prefix] = prefill_meta
                total_need_load += 1
            else:
                # Prefill request: save KV to files.
                # Use get() not pop() because build_connector_meta is
                # called once per CP rank — all ranks need to see the
                # same prefill request. Clean up after last rank.
                kv_params = self._prefill_requests.get(
                    new_req.req_id, None
                )
                # Fallback: if get_num_new_matched_tokens was never
                # called for this request (e.g., it was deferred in the
                # waiting queue across scheduling rounds), retrieve
                # kv_transfer_params directly from the request object
                # and register it now.
                if kv_params is None and new_req.sampling_params is not None:
                    fallback_kv = (new_req.sampling_params.extra_args or {}).get(
                        "kv_transfer_params", None)
                    if fallback_kv and fallback_kv.get("do_remote_decode"):
                        kv_params = fallback_kv
                        self._prefill_requests[new_req.req_id] = kv_params
                        logger.warning(
                            "build_connector_meta: late-registered "
                            "prefill req=%s via sampling_params fallback",
                            new_req.req_id,
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

        # Check for prefill continuation chunks in running requests.
        # Running requests appear in num_scheduled_tokens but NOT in
        # scheduled_new_reqs or scheduled_cached_reqs.
        processed_new = {nr.req_id for nr in scheduler_output.scheduled_new_reqs}
        for req_id, num_sched in scheduler_output.num_scheduled_tokens.items():
            if req_id in processed_new:
                continue  # Already handled in new_reqs loop
            prefill_kv = self._prefill_requests.get(req_id)
            if prefill_kv and prefill_kv.get("do_remote_decode") and num_sched > 1:
                # Skip requests not assigned to this cp_rank
                if scheduler_output.cp_rank_scheduled_tokens.get(req_id, 0) == 0:
                    continue
                # Running prefill continuation chunk
                pd_prefix = prefill_kv.get("pd_request_prefix", "")
                cp_size = scheduler_output.cp_rank_scheduled_tokens.get(
                    req_id, 1
                )
                if cp_size > 1:
                    import numpy as np
                    padded = int(
                        np.ceil(num_sched / (2 * cp_size)) * (2 * cp_size)
                    )
                    tpr = padded // cp_size
                else:
                    tpr = None
                meta.add_request(
                    req_id=req_id,
                    token_ids=[],
                    block_ids=[],  # block_ids not needed for pre-mask save
                    block_size=self._block_size,
                    is_store=True,
                    pd_request_prefix=pd_prefix,
                    num_tokens_override=tpr,
                )

        # Handle cached/resumed requests (decode load)
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached_reqs.req_ids):
            resumed = req_id in cached_reqs.resumed_req_ids

            # Decode load
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
            prefill_meta = self._completed_prefills.get(pd_prefix)
            if prefill_meta:
                meta.ipc_prefill_metas[pd_prefix] = prefill_meta
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
        block_ids,
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

            # Extract per-rank block_ids for IPC KV transfer.
            # block_ids from CrossDPKVCacheManager.get_block_ids():
            #   list[tuple[list[int], ...]], one tuple per CP rank
            # For single kv_cache_group: block_ids[i][0] = list[int]
            per_rank_block_ids = []
            if isinstance(block_ids, list) and len(block_ids) > 0:
                first = block_ids[0]
                if isinstance(first, (list, tuple)) and len(first) > 0 and isinstance(first[0], (list, tuple)):
                    # New format: list[tuple[list[int], ...]]
                    per_rank_block_ids = [list(rank_blocks[0]) for rank_blocks in block_ids]
                else:
                    # Legacy format: single list[int] or tuple[list[int], ...]
                    if isinstance(first, (list, tuple)):
                        per_rank_block_ids = [list(first)]
                    else:
                        per_rank_block_ids = [list(block_ids)]

            meta = {
                "completed": True,
                "cp_world_size": actual_cp_count,
                "num_prompt_tokens": len(request.prompt_token_ids),
                "block_size": self._block_size,
                "pd_request_prefix": prefix,
                "per_rank_block_ids": per_rank_block_ids,
                "interleave_size": self._interleave_size,
                "prefill_req_id": request.request_id,
            }

            # Store in memory (no file I/O)
            import time as _time
            self._completed_prefills[prefix] = meta
            meta["_finish_time_ms"] = _time.monotonic() * 1000
            # Clean up prefill tracking (all chunks done)
            self._prefill_requests.pop(request.request_id, None)

            logger.info(
                "Prefill finished for prefix=%s "
                "(num_prompt_tokens=%d, cp_world_size=%d, "
                "per_rank_blocks=%s)",
                prefix,
                len(request.prompt_token_ids),
                actual_cp_count,
                [len(b) for b in per_rank_block_ids],
            )

            return_params = {
                "pd_request_prefix": prefix,
                "cp_world_size": actual_cp_count,
                "num_prompt_tokens": len(request.prompt_token_ids),
                "prompt_token_ids": list(request.prompt_token_ids),
                "per_rank_block_ids": per_rank_block_ids,
                "interleave_size": self._interleave_size,
            }
            # IPC mode: delay freeing prefill blocks until decode reads them.
            # The worker's get_finished() returns finished_sending
            # after start_load_kv completes IPC copy.
            delay_free = actual_cp_count > 1
            if delay_free:
                self._ipc_delayed_prefill_ids[prefix] = request.request_id
                return_params["prefill_req_id"] = request.request_id
                logger.info(
                    "Delaying block free for prefix=%s req=%s",
                    prefix, request.request_id,
                )
            return delay_free, return_params

        if kv_params.get("do_remote_prefill"):
            # Decode finished: clean up in-memory metadata
            prefix = kv_params.get("pd_request_prefix", "")
            self._completed_prefills.pop(prefix, None)
            self._prefill_requests.pop(request.request_id, None)
            return False, None

        return False, None

    # ==============================
    # Worker-side methods
    # ==============================

    def start_load_kv(
        self, forward_context: "ForwardContext", **kwargs: Any
    ) -> None:
        """Load KV cache for decode requests."""
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, LocalPDConnectorMetadata):
            return

        for req_meta in metadata.requests:
            if req_meta.is_store:
                continue

            prefix = req_meta.pd_request_prefix

            if self._ipc_initialized:
                self._start_load_kv_ipc(
                    req_meta, forward_context, prefix
                )
            else:
                self._start_load_kv_legacy(
                    req_meta, forward_context, prefix
                )

    def _start_load_kv_ipc(
        self,
        req_meta: LocalPDReqMeta,
        forward_context: "ForwardContext",
        prefix: str,
    ) -> None:
        """Load KV via CUDA IPC from remote ranks' paged buffers."""
        import time as _time
        t0 = _time.monotonic()

        # Get IPC metadata from connector metadata (passed from scheduler)
        connector_meta = self._get_connector_metadata()
        if not isinstance(connector_meta, LocalPDConnectorMetadata):
            logger.error("No connector metadata for IPC load")
            return

        meta = connector_meta.ipc_prefill_metas.get(prefix)
        if meta is None:
            logger.error("No IPC prefill metadata for prefix=%s", prefix)
            return

        per_rank_block_ids = meta.get("per_rank_block_ids")
        if not per_rank_block_ids:
            logger.error("No per_rank_block_ids for prefix=%s", prefix)
            return

        cp_world_size = meta.get("cp_world_size", self._cp_world_size)
        block_size = self._block_size
        interleave_size = meta.get("interleave_size", self._interleave_size)

        from vllm.distributed.parallel_state import get_dycp_group
        my_rank = get_dycp_group().rank_in_group

        dst_slot_mapping = req_meta.slot_mapping
        actual_tokens = dst_slot_mapping.shape[0]

        # Compute interleave mapping: which src_rank owns each position
        # and what slot in that rank's paged buffer holds the KV.
        # Formula from block_table.py:238-250.
        positions = np.arange(actual_tokens)
        virtual_block_size = block_size * cp_world_size
        virtual_block_offsets = positions % virtual_block_size

        owning_ranks = (
            virtual_block_offsets // interleave_size % cp_world_size
        )
        # Local block offset within the owning rank's paged buffer
        local_block_offsets = (
            virtual_block_offsets
            // (cp_world_size * interleave_size)
            * interleave_size
            + virtual_block_offsets % interleave_size
        )
        # Which block (index) within the rank's allocation
        block_indices = positions // virtual_block_size

        # Group block copies by src_rank to minimize cudaMemcpy calls.
        # For each src_rank, identify unique src_block -> dst_block pairs.
        # Copy entire blocks (block_size * kv_dim * elem_size bytes each).
        per_rank_copies: dict[int, list[tuple[int, int]]] = {}
        for src_rank in range(cp_world_size):
            rank_mask = owning_ranks == src_rank
            if not np.any(rank_mask):
                continue

            rank_positions = np.where(rank_mask)[0]
            src_block_ids_arr = np.array(per_rank_block_ids[src_rank])

            rank_block_indices = block_indices[rank_positions]
            rank_block_indices = np.clip(
                rank_block_indices, 0,
                max(len(src_block_ids_arr) - 1, 0),
            )
            src_block_ids_actual = src_block_ids_arr[rank_block_indices]
            dst_slots = dst_slot_mapping[rank_positions].numpy()
            dst_block_ids_actual = dst_slots // block_size

            # Unique (src_block, dst_block) pairs
            pairs = set(zip(
                src_block_ids_actual.tolist(),
                dst_block_ids_actual.tolist(),
            ))
            per_rank_copies[src_rank] = sorted(pairs)

        is_mla = None
        layers_injected = 0
        cudamemcpy_fn = self._cuda_lib.funcs["cudaMemcpyAsync"]
        ipc_stream = self._ipc_stream

        for layer_name, layer in forward_context.no_compile_layers.items():
            kv_cache_attr = getattr(layer, "kv_cache", None)
            if kv_cache_attr is None:
                continue
            kv_cache_layer = kv_cache_attr[forward_context.virtual_engine]

            if is_mla is None:
                is_mla = kv_cache_layer.dim() == 3

            # Compute bytes per block for this layer
            if is_mla:
                # Shape: [num_pages, page_size, kv_dim]
                kv_dim = kv_cache_layer.shape[2]
                elem_size = kv_cache_layer.element_size()
                slot_bytes = kv_dim * elem_size
            else:
                # Shape: [2, num_pages, page_size, kv_dim]
                kv_dim = kv_cache_layer.shape[3]
                elem_size = kv_cache_layer.element_size()
                slot_bytes = kv_dim * elem_size
                # For non-MLA, need to handle 2 (K+V) separately

            block_bytes = block_size * slot_bytes
            local_base_ptr = kv_cache_layer.data_ptr()

            for src_rank, block_pairs in per_rank_copies.items():
                if src_rank == my_rank:
                    # Local rank: copy within same paged buffer
                    src_base_ptr = local_base_ptr
                else:
                    ipc_info = self._remote_ipc_info[src_rank][layer_name]
                    remote_ptr = ipc_info[0]
                    src_base_ptr = remote_ptr.value

                for src_block, dst_block in block_pairs:
                    if is_mla:
                        src_offset = src_block * block_bytes
                        dst_offset = dst_block * block_bytes
                        self._cuda_lib.CUDART_CHECK(
                            cudamemcpy_fn(
                                ctypes.c_void_p(
                                    local_base_ptr + dst_offset
                                ),
                                ctypes.c_void_p(
                                    src_base_ptr + src_offset
                                ),
                                ctypes.c_size_t(block_bytes),
                                ctypes.c_int(4),  # cudaMemcpyDefault
                                ipc_stream,
                            )
                        )
                    else:
                        # Non-MLA: [2, num_pages, page_size, kv_dim]
                        # Copy K and V separately
                        num_pages = kv_cache_layer.shape[1]
                        plane_bytes = (
                            num_pages * block_size * slot_bytes
                        )
                        for plane in range(2):
                            src_offset = (
                                plane * plane_bytes
                                + src_block * block_bytes
                            )
                            dst_offset = (
                                plane * plane_bytes
                                + dst_block * block_bytes
                            )
                            self._cuda_lib.CUDART_CHECK(
                                cudamemcpy_fn(
                                    ctypes.c_void_p(
                                        local_base_ptr + dst_offset
                                    ),
                                    ctypes.c_void_p(
                                        src_base_ptr + src_offset
                                    ),
                                    ctypes.c_size_t(block_bytes),
                                    ctypes.c_int(4),
                                    ipc_stream,
                                )
                            )

            layers_injected += 1

        # Record event on IPC stream to track async completion.
        # get_finished() polls cudaEventQuery to detect when done.
        event_ptr = ctypes.c_void_p()
        self._cuda_lib.CUDART_CHECK(
            self._cuda_lib.funcs["cudaEventCreate"](
                ctypes.byref(event_ptr)
            )
        )
        self._cuda_lib.CUDART_CHECK(
            self._cuda_lib.funcs["cudaEventRecord"](
                event_ptr, self._ipc_stream
            )
        )

        # Track: decode_req_id → (event, prefill_req_id)
        prefill_req_id = meta.get("prefill_req_id")
        decode_req_id = req_meta.req_id
        self._ipc_pending_events[decode_req_id] = (
            event_ptr, prefill_req_id,
        )

        elapsed = (_time.monotonic() - t0) * 1000
        logger.info(
            "IPC KV async launched for prefix=%s: %d layers, %d tokens "
            "from %d ranks in %.1fms (decode=%s, prefill=%s)",
            prefix, layers_injected, actual_tokens,
            cp_world_size, elapsed, decode_req_id, prefill_req_id,
        )

    def _start_load_kv_legacy(
        self,
        req_meta: LocalPDReqMeta,
        forward_context: "ForwardContext",
        prefix: str,
    ) -> None:
        """Load KV from _gpu_kv_buffer (legacy all-gather path)."""
        import time as _time

        kv_buf = self._gpu_kv_buffer.get(prefix)
        if kv_buf is None:
            logger.error("No GPU KV buffer for prefix=%s", prefix)
            return

        actual_tokens = req_meta.slot_mapping.shape[0]
        _t0 = _time.monotonic()

        is_mla = None
        sm = req_meta.slot_mapping
        layers_injected = 0
        for layer_name, layer in forward_context.no_compile_layers.items():
            kv_cache_attr = getattr(layer, "kv_cache", None)
            if kv_cache_attr is None:
                continue
            kv_cache_layer = kv_cache_attr[forward_context.virtual_engine]

            if is_mla is None:
                is_mla = kv_cache_layer.dim() == 3

            full_kv = kv_buf.get(layer_name)
            if full_kv is None:
                continue

            if is_mla:
                num_pages = kv_cache_layer.shape[0]
                page_size = kv_cache_layer.shape[1]
                flat = kv_cache_layer.reshape(num_pages * page_size, -1)
                flat[sm] = full_kv[:actual_tokens]
            else:
                num_pages = kv_cache_layer.shape[1]
                page_size = kv_cache_layer.shape[2]
                flat = kv_cache_layer.reshape(2, num_pages * page_size, -1)
                flat[:, sm] = full_kv[:, :actual_tokens]
            layers_injected += 1

        del self._gpu_kv_buffer[prefix]

        _elapsed = (_time.monotonic() - _t0) * 1000
        logger.info(
            "KV injected (legacy) for prefix=%s: %d layers in %.1fms",
            prefix, layers_injected, _elapsed,
        )

    def wait_for_layer_load(self, layer_name: str) -> None:
        """GPU-level sync: make default stream wait for IPC event.

        Uses cudaStreamWaitEvent — no CPU blocking. The GPU naturally
        orders: IPC copies complete → then attention kernels run.
        """
        if not self._ipc_pending_events:
            return
        if getattr(self, '_ipc_gpu_synced', False):
            return
        # On first call, insert GPU-level wait for ALL pending IPC events.
        # After this, default stream won't execute until IPC stream finishes.
        for decode_req_id, (event, _) in self._ipc_pending_events.items():
            default_stream = torch.cuda.current_stream().cuda_stream
            self._cuda_lib.CUDART_CHECK(
                self._cuda_lib.funcs["cudaStreamWaitEvent"](
                    ctypes.c_void_p(default_stream),
                    event,
                    ctypes.c_uint(0),
                )
            )
        # Mark as synced — subsequent layer calls are no-op
        self._ipc_gpu_synced = True

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> None:
        """Save KV cache for prefill requests.

        IPC mode: KV is already in each rank's paged buffer via the
        attention kernel; decode reads via IPC. For multi-rank CP
        requests, no extraction is needed. For single-rank requests,
        extract KV from the paged buffer into _gpu_kv_buffer.
        """
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, LocalPDConnectorMetadata):
            return
        if not metadata.has_store_requests:
            return

        # In IPC mode with multi-rank CP, KV is already in paged buffers.
        # No extraction needed — decode reads directly via IPC.
        cp_size = getattr(attn_metadata, "num_dycp_reqs", 0)
        if self._ipc_initialized and self._cp_world_size > 1 and cp_size > 0:
            return

        is_mla = isinstance(attn_metadata, MLACommonMetadata)

        for req_meta in metadata.requests:
            if not req_meta.is_store:
                continue

            prefix = req_meta.pd_request_prefix

            actual_sm = getattr(attn_metadata, "slot_mapping", None)
            if actual_sm is not None:
                actual_sm = actual_sm.flatten()
                valid = actual_sm >= 0
                sm = actual_sm[valid]
            else:
                sm = req_meta.slot_mapping
            if sm.shape[0] == 0:
                continue
            kv_data = _extract_kv_from_layer(kv_layer, sm, is_mla)
            existing = self._gpu_kv_buffer.get(prefix, {}).get(
                layer_name
            )
            if existing is not None:
                if is_mla:
                    kv_data = torch.cat([existing, kv_data], dim=0)
                else:
                    kv_data = torch.cat([existing, kv_data], dim=1)
            self._gpu_kv_buffer.setdefault(prefix, {})[layer_name] = (
                kv_data.detach()
            )

    def wait_for_save(self):
        """In IPC mode: no-op. KV stays in each rank's paged buffer."""
        if not self._pending_local_kv:
            logger.info("wait_for_save: IPC mode, no-op (0ms)")
            return

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Poll CUDA events for IPC completion.

        finished_sending: prefill req_ids whose IPC copy is confirmed
            done (blocks can be freed).
        finished_recving: decode req_ids done loading (sync mode: all).
        """
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, LocalPDConnectorMetadata):
            return None, None

        finished_sending: set[str] = set()

        # Poll each pending IPC event (non-blocking cudaEventQuery)
        if self._cuda_lib and self._ipc_pending_events:
            event_query_fn = self._cuda_lib.funcs["cudaEventQuery"]
            completed: list[str] = []
            for decode_req_id, (event, prefill_req_id) in (
                self._ipc_pending_events.items()
            ):
                result = event_query_fn(event)
                if result == 0:  # cudaSuccess
                    completed.append(decode_req_id)
                    if prefill_req_id:
                        finished_sending.add(prefill_req_id)
            for did in completed:
                del self._ipc_pending_events[did]
            # Reset gpu sync flag for next batch
            if completed:
                self._ipc_gpu_synced = False
                logger.info(
                    "get_finished: IPC done, freeing prefill blocks %s",
                    finished_sending,
                )

        # finished_recving: sync mode, all load requests are done
        load_req_ids = {
            r.req_id for r in metadata.requests if not r.is_store
        }

        return (
            finished_sending if finished_sending else None,
            load_req_ids if load_req_ids else None,
        )
