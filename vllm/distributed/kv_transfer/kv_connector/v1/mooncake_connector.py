# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import msgspec
import numpy as np
import torch
import zmq
import zmq.asyncio

from vllm import envs
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.selector import get_attn_backend
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import TpKVTopology
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

try:
    from mooncake.engine import TransferEngine
except ImportError as e:
    raise ImportError(
        "Please install mooncake by following the instructions at "
        "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
        "to run VLLM with MooncakeTransferEngine."
    ) from e

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

EngineId = str
ReqId = str

TRANS_DONE = b"trans_done"
TRANS_ERROR = b"trans_error"

logger = init_logger(__name__)


class MooncakeAgentMetadata(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    # required for @cached_property.
    dict=True,
):
    remote_hostname: str
    remote_port: int
    request_ids: list[ReqId]
    kv_caches_base_addr: list[int]
    block_ids: list[list[int]]
    # CP-aware PD fields (DyCP support)
    cp_rank: int = 0
    cp_world_size: int = 1
    cp_interleave_size: int = 64
    cp_block_size: int = 16
    num_tokens: list[int] = []
    # Sender's TP size so receiver can compute tp_head_offset for GQA.
    tp_size: int = 0


@dataclass
class RecvReqMeta:
    local_block_ids: list[int]
    remote_host: str
    remote_port: int
    # P's request_id for handshake (PD separation: P and D have different IDs)
    remote_request_id: str = ""
    # CP topology info from prefill side
    prefill_cp_world_size: int = 1
    prefill_cp_interleave_size: int = 64
    prefill_tp_size: int = 1
    # Which P ranks actually hold data for this request
    remote_dycp_ranks: list[int] | None = None
    # Decode-side CP info (this rank)
    decode_cp_rank: int = 0
    decode_cp_world_size: int = 1
    decode_cp_interleave_size: int = 64
    num_tokens: int = 0


@dataclass
class SendBlockMeta:
    local_block_ids: list[int]
    ready: threading.Event
    expire_time: float = float("inf")
    # CP-aware PD: M decode ranks request the same KV.
    # Only delete after all receivers complete.
    expected_receivers: int = 1
    completed_receivers: int = 0


@dataclass
class SendReqMeta:
    reqs: dict[ReqId, SendBlockMeta]
    lock: threading.Lock


@dataclass
class FinishedSendReqSet:
    set: set[ReqId]
    lock: threading.Lock


@dataclass
class FinishedReceiveReqSet:
    set: set[ReqId]
    lock: asyncio.Lock


class MooncakeConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.reqs_to_recv: dict[ReqId, RecvReqMeta] = {}
        self.reqs_to_send: dict[ReqId, list[int]] = {}

    def add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
        load_remote_cache: bool = True,
        num_tokens: int = 0,
        decode_cp_rank: int = 0,
        decode_cp_world_size: int = 1,
        decode_cp_interleave_size: int = 64,
    ):
        if load_remote_cache:
            self.reqs_to_recv[request_id] = RecvReqMeta(
                local_block_ids=local_block_ids,
                remote_host=kv_transfer_params["remote_host"],
                remote_port=kv_transfer_params["remote_port"],
                remote_request_id=kv_transfer_params.get(
                    "remote_request_id", request_id
                ),
                prefill_cp_world_size=kv_transfer_params.get(
                    "prefill_cp_world_size", 1
                ),
                prefill_cp_interleave_size=kv_transfer_params.get(
                    "prefill_cp_interleave_size", 64
                ),
                prefill_tp_size=kv_transfer_params.get(
                    "prefill_tp_size", 1
                ),
                remote_dycp_ranks=kv_transfer_params.get(
                    "remote_dycp_ranks", None
                ),
                decode_cp_rank=decode_cp_rank,
                decode_cp_world_size=decode_cp_world_size,
                decode_cp_interleave_size=decode_cp_interleave_size,
                num_tokens=num_tokens,
            )
        else:
            self.reqs_to_send[request_id] = local_block_ids


class MooncakeConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: MooncakeConnectorScheduler | None = (
                MooncakeConnectorScheduler(vllm_config, self.engine_id)
            )
            self.connector_worker: MooncakeConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeConnectorWorker(vllm_config, self.engine_id)

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def get_req_id_to_cp_size(self) -> dict[str, int] | None:
        """Return per-request CP size for aggregate_domain()."""
        if self.connector_worker is not None:
            sizes = self.connector_worker._req_cp_sizes
            return sizes if sizes else None
        return None

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """MooncakeConnector does not do layerwise saving."""
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> None:
        """MooncakeConnector does not save explicitly."""
        pass

    def wait_for_save(self):
        pass


class MooncakeConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.vllm_config = vllm_config
        self.engine_id: EngineId = engine_id
        self.side_channel_host = get_ip()
        self.side_channel_port = get_mooncake_side_channel_port(vllm_config)

        assert vllm_config.kv_transfer_config
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        logger.info("Initializing Mooncake Transfer Engine Scheduler %s", engine_id)

        # Domain / DyCP config.
        self.dp_per_domain = getattr(
            vllm_config.parallel_config, 'dp_per_domain', 1
        ) or 1
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.cp_interleave_size = getattr(
            vllm_config.parallel_config,
            'cp_kv_cache_interleave_size', 64
        ) or 64

        # Domain port base: when dp_per_domain > 1, use domain-level port
        # so all DP ranks within a domain share a port namespace.
        if self.dp_per_domain > 1:
            domain_rank = getattr(
                vllm_config.parallel_config, 'domain_parallel_rank', 0
            )
            kv_port = vllm_config.kv_transfer_config.kv_port or 20002
            self.domain_port_base = (
                kv_port
                + domain_rank * self.tp_size * self.dp_per_domain
            )
        else:
            self.domain_port_base = self.side_channel_port

        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        # Key: (req_id, target_rank). rank=-1 means any rank (normal scheduler).
        self._reqs_need_recv: dict[tuple[ReqId, int], tuple[Request, list[int]]] = {}
        self._reqs_need_send: dict[ReqId, list[int]] = {}
        # Per-rank block_ids for SEND filtering: {req_id: {rank: block_ids}}
        self._send_per_rank_blocks: dict[ReqId, dict[int, list[int]]] = {}

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        """
        For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the
              external KV cache beyond what is already computed.
            * true if the external KV cache tokens will be loaded
              asynchronously (between scheduler steps).
        """

        params = request.kv_transfer_params

        if params is not None and params.get("do_remote_prefill"):
            token_ids = request.prompt_token_ids or []
            count = len(token_ids) - num_computed_tokens
            logger.info(
                "[PD] get_num_new_matched_tokens: req=%s, "
                "num_computed=%d, prompt_len=%d, external_count=%d",
                request.request_id, num_computed_tokens,
                len(token_ids), max(count, 0),
            )
            if count > 0:
                return count, True

        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks, num_external_tokens: int
    ):
        params = request.kv_transfer_params
        logger.info(
            "[PD] update_state_after_alloc: req=%s, "
            "num_external_tokens=%s, blocks_type=%s, "
            "cp_ranks=%s, kv_transfer_params=%s",
            request.request_id,
            num_external_tokens,
            type(blocks).__name__,
            getattr(request, 'cp_ranks', None),
            params,
        )

        if not params:
            return

        if params.get("do_remote_prefill"):
            assert self.kv_role != "kv_producer"
            if all(p in params for p in ("remote_host", "remote_port")):
                cp_ranks = getattr(request, 'cp_ranks', None)
                if isinstance(blocks, list):
                    ranks = cp_ranks if cp_ranks else list(range(len(blocks)))
                    for rank, rank_blocks in zip(ranks, blocks):
                        block_ids = (
                            rank_blocks.get_unhashed_block_ids()
                            if num_external_tokens > 0 else []
                        )
                        self._reqs_need_recv[(request.request_id, rank)] = (
                            request, block_ids
                        )
                        logger.info(
                            "[PD] recv entry added: req=%s, rank=%d, "
                            "num_blocks=%d, block_ids=%s",
                            request.request_id, rank,
                            len(block_ids), block_ids[:8],
                        )
                else:
                    local_block_ids = (
                        blocks.get_unhashed_block_ids()
                        if num_external_tokens > 0 else []
                    )
                    self._reqs_need_recv[(request.request_id, -1)] = (
                        request, local_block_ids
                    )
                    logger.info(
                        "[PD] recv entry added: req=%s, rank=-1, "
                        "num_blocks=%d",
                        request.request_id, len(local_block_ids),
                    )
            else:
                logger.warning(
                    "Got invalid KVTransferParams: %s. This "
                    "request will not utilize KVTransfer",
                    params,
                )
            params["do_remote_prefill"] = False

        elif params.get("do_remote_decode"):
            self._reqs_need_send[request.request_id] = []
            logger.info(
                "[PD] send entry added (empty): req=%s",
                request.request_id,
            )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MooncakeConnectorMetadata()
        current_rank = (
            scheduler_output.cp_rank
            if hasattr(scheduler_output, 'cp_rank')
            else -1
        )

        if self.kv_role != "kv_producer":
            keys_to_remove = []
            for (req_id, rank), (req, block_ids) in self._reqs_need_recv.items():
                if rank != -1 and rank != current_rank:
                    continue
                assert req.kv_transfer_params is not None
                num_tokens = len(req.prompt_token_ids or [])
                cp_ranks = getattr(req, 'cp_ranks', None)
                if cp_ranks and len(cp_ranks) > 1:
                    d_cp_ws = len(cp_ranks)
                    d_cp_rank = (cp_ranks.index(current_rank)
                                 if current_rank in cp_ranks else 0)
                else:
                    d_cp_ws = 1
                    d_cp_rank = 0
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                    num_tokens=num_tokens,
                    decode_cp_rank=d_cp_rank,
                    decode_cp_world_size=d_cp_ws,
                    decode_cp_interleave_size=self.cp_interleave_size,
                )
                keys_to_remove.append((req_id, rank))
                logger.info(
                    "[PD] build_connector_meta RECV: req=%s, "
                    "current_rank=%d, target_rank=%d, "
                    "num_blocks=%d, num_tokens=%d, "
                    "d_cp_rank=%d, d_cp_ws=%d, cp_interleave=%d",
                    req_id, current_rank, rank,
                    len(block_ids), num_tokens,
                    d_cp_rank, d_cp_ws, self.cp_interleave_size,
                )
            for key in keys_to_remove:
                del self._reqs_need_recv[key]

        if self.kv_role != "kv_consumer":
            sent_ready_keys: list[str] = []
            for req_id, block_ids in self._reqs_need_send.items():
                per_rank = self._send_per_rank_blocks.get(req_id)
                if per_rank is not None:
                    if current_rank >= 0 and current_rank not in per_rank:
                        continue
                    rank_block_ids = per_rank.get(current_rank, block_ids)
                else:
                    rank_block_ids = block_ids
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=rank_block_ids,
                    kv_transfer_params={},
                    load_remote_cache=False,
                )
                logger.info(
                    "[PD] build_connector_meta SEND: req=%s, "
                    "current_rank=%d, num_blocks=%d, "
                    "per_rank_filtered=%s",
                    req_id, current_rank, len(rank_block_ids),
                    per_rank is not None,
                )
                if block_ids:
                    sent_ready_keys.append(req_id)
            for req_id in sent_ready_keys:
                del self._reqs_need_send[req_id]
                self._send_per_rank_blocks.pop(req_id, None)

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """

        params = request.kv_transfer_params
        logger.info(
            "[PD] request_finished: req=%s, status=%s, "
            "num_block_ids=%d, kv_transfer_params=%s",
            request.request_id, request.status,
            len(block_ids), params,
        )
        if not params:
            return False, None

        if params.get("do_remote_prefill"):
            # If do_remote_prefill is still True when the request is finished,
            # update_state_after_alloc must not have been called (the request
            # must have been aborted before it was scheduled).
            # To avoid stranding the prefill blocks in the prefill instance,
            # we must add empty block_ids to _reqs_need_recv so that our
            # worker side will notify and free blocks in the prefill instance.
            assert self.kv_role != "kv_producer"
            self._reqs_need_recv[(request.request_id, -1)] = (request, [])
            params["do_remote_prefill"] = False
            return False, None

        if (
            not params.get("do_remote_decode")
            or request.status != RequestStatus.FINISHED_LENGTH_CAPPED
        ):
            return False, None

        assert self.kv_role != "kv_consumer"

        # TODO: check whether block_ids actually ever be 0. If not we could
        # remove the conditional below
        delay_free_blocks = len(block_ids) > 0

        if delay_free_blocks:
            self._reqs_need_send[request.request_id] = block_ids

        # Use domain_port_base when dp_per_domain > 1.
        remote_port = self.domain_port_base

        cp_ranks = getattr(request, 'cp_ranks', None)
        p_cp_ws = len(cp_ranks) if cp_ranks else self.dp_per_domain

        return delay_free_blocks, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_host=self.side_channel_host,
            remote_port=remote_port,
            remote_request_id=request.request_id,
            prefill_cp_world_size=p_cp_ws,
            prefill_cp_interleave_size=self.cp_interleave_size,
            prefill_tp_size=self.tp_size,
            remote_dycp_ranks=cp_ranks or [],
        )


class MooncakeConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        logger.info("Initializing Mooncake Transfer Engine worker %s", engine_id)

        self.vllm_config = vllm_config

        self.engine = TransferEngine()
        self.hostname = get_ip()
        ret_value = self.engine.initialize(self.hostname, "P2PHANDSHAKE", "rdma", "")
        if ret_value != 0:
            raise RuntimeError("Mooncake Transfer Engine initialization failed.")

        self.rpc_port = self.engine.get_rpc_port()

        logger.debug(
            "Mooncake Transfer Engine initialized at %s:%d",
            self.hostname,
            self.rpc_port,
        )

        # Mooncake handshake port.
        self.side_channel_port: int = get_mooncake_side_channel_port(vllm_config)

        self.engine_id: EngineId = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.world_size = get_tensor_model_parallel_world_size()
        self.tp_group = get_tp_group()
        self.num_blocks = 0

        assert vllm_config.kv_transfer_config
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.num_workers = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "num_workers", 10
        )

        extra_cfg = vllm_config.kv_transfer_config.kv_connector_extra_config
        if self.kv_role == "kv_producer":
            self.remote_tp_size = extra_cfg.get("decode", {}).get(
                "tp_size", self.world_size)
        else:
            self.remote_tp_size = extra_cfg.get("prefill", {}).get(
                "tp_size", self.world_size)

        self.kv_caches_base_addr: list[int] = []
        self.device_kv_caches: dict[str, torch.Tensor] = {}
        self.reqs_need_send: SendReqMeta = SendReqMeta(reqs={}, lock=threading.Lock())

        # For kv_both, we will act both prefiller and decoder.
        if self.kv_role != "kv_consumer":
            # Background thread for sending kvcaches to D.
            self._mooncake_sender_t: threading.Thread | None = None
            # Background thread for processing new sending requests.
            self._sender_executor = ThreadPoolExecutor(
                max_workers=self.num_workers, thread_name_prefix="vllm-mooncake-sender"
            )
            logger.debug(
                "Mooncake Prefiller: use %d workers to send kvcaches", self.num_workers
            )
        if self.kv_role != "kv_producer":
            self.receiver_loop = asyncio.new_event_loop()
            self._mooncake_receiver_t = threading.Thread(
                target=self._receiver_loop, args=(self.receiver_loop,), daemon=True
            )
            self._mooncake_receiver_t.start()
            logger.debug("Mooncake Decoder: start receiver thread")

        self.finished_sending_reqs: FinishedSendReqSet = FinishedSendReqSet(
            set(), threading.Lock()
        )
        self.finished_recving_reqs: FinishedReceiveReqSet = FinishedReceiveReqSet(
            set(), asyncio.Lock()
        )
        self._recv_pending_counts: dict[str, int] = {}
        self._req_cp_sizes: dict[str, int] = {}

        self.block_size = vllm_config.cache_config.block_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.use_mla = self.model_config.use_mla

        # CP topology for DyCP PD scatter.
        self._local_cp_world_size = getattr(
            vllm_config.parallel_config, 'dp_per_domain', 1
        ) or 1
        self._local_cp_interleave_size = getattr(
            vllm_config.parallel_config,
            'cp_kv_cache_interleave_size', 64
        ) or 64
        self._local_cp_rank = 0
        if self._local_cp_world_size > 1:
            try:
                from vllm.distributed.parallel_state import get_dycp_group
                self._local_cp_rank = get_dycp_group().rank_in_group
            except (AssertionError, RuntimeError):
                self._local_cp_rank = (
                    vllm_config.parallel_config.data_parallel_rank
                    % self._local_cp_world_size
                )

        # Domain port base for sender thread.
        if self._local_cp_world_size > 1:
            domain_rank = getattr(
                vllm_config.parallel_config, 'domain_parallel_rank', 0
            )
            kv_port = (
                vllm_config.kv_transfer_config.kv_port
                if vllm_config.kv_transfer_config
                else 20002
            ) or 20002
            self._domain_port_base = (
                kv_port
                + domain_rank
                * self.world_size
                * self._local_cp_world_size
            )
        else:
            self._domain_port_base = self.side_channel_port

        backend = get_attn_backend(
            self.model_config.get_head_size(),
            self.model_config.dtype,
            self.cache_config.cache_dtype,
            self.block_size,
            use_mla=self.use_mla,
        )
        self.backend_name = backend.get_name()
        self.kv_cache_layout = get_kv_cache_layout()
        logger.debug("Detected attention backend %s", self.backend_name)
        logger.debug("Detected kv cache layout %s", self.kv_cache_layout)

        self._tp_size: dict[EngineId, int] = {self.engine_id: self.world_size}
        self._block_size: dict[EngineId, int] = {self.engine_id: self.block_size}
        self.kv_topo = TpKVTopology(
            tp_rank=self.tp_rank,
            engine_id=self.engine_id,
            remote_tp_size=self._tp_size,  # shared state
            remote_block_size=self._block_size,  # shared state
            is_mla=self.use_mla,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backend=backend,
        )
        self._use_pallas = self.kv_topo._use_pallas

        self.zmq_ctx = zmq.Context()
        self.async_zmq_ctx = zmq.asyncio.Context()
        self._encoder = msgspec.msgpack.Encoder()
        self._decoder = msgspec.msgpack.Decoder(MooncakeAgentMetadata)

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        """Cleanup background threads on destruction."""
        self.zmq_ctx.term()
        self.async_zmq_ctx.term()
        if self.kv_role != "kv_consumer":
            self._sender_executor.shutdown(wait=False)
            if self._mooncake_sender_t:
                self._mooncake_sender_t.join()
        if self.kv_role != "kv_producer" and self.receiver_loop.is_running():
            self.receiver_loop.call_soon_threadsafe(self.receiver_loop.stop)
            self._mooncake_receiver_t.join()

    def _receiver_loop(self, loop: asyncio.AbstractEventLoop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def _mooncake_sender(
        self, ready_event: threading.Event, base_port: int, tp_rank: int
    ):
        """
        Background thread that listens for Mooncake requests, dispatches them
        to a thread pool, and sends acknowledgments upon completion.
        """

        frontend_path = make_zmq_path("tcp", self.hostname, base_port + tp_rank)
        frontend = make_zmq_socket(self.zmq_ctx, frontend_path, zmq.ROUTER)
        logger.debug("Mooncake sender starting listening on path: %s", frontend_path)

        backend_path = make_zmq_path("inproc", str(uuid.uuid4()))
        backend = make_zmq_socket(self.zmq_ctx, backend_path, zmq.PULL)

        poller = zmq.Poller()
        poller.register(frontend, zmq.POLLIN)
        poller.register(backend, zmq.POLLIN)

        ready_event.set()

        try:
            while True:
                sockets = dict(poller.poll())

                if frontend in sockets:
                    identity, _, metadata_bytes = frontend.recv_multipart()
                    self._sender_executor.submit(
                        self._sender_worker,
                        identity,
                        metadata_bytes,
                        backend_path,
                    )

                if backend in sockets:
                    identity, status = backend.recv_multipart()
                    frontend.send_multipart((identity, b"", status))

        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake sender thread.")
        except Exception as e:
            logger.error("Error in Mooncake sender thread: %s. Exiting thread.", str(e))
        finally:
            frontend.close()
            backend.close()

    def _sender_worker(
        self, identity: bytes, metadata_bytes: bytes, worker_channel_path: str
    ):
        status = TRANS_ERROR

        try:
            metadata = self._decoder.decode(metadata_bytes)
            self.send_kv_to_decode(metadata)
            status = TRANS_DONE
        except Exception as e:
            logger.error("Error processing Mooncake handshake: %s", e)
        finally:
            pusher = make_zmq_socket(self.zmq_ctx, worker_channel_path, zmq.PUSH)
            try:
                pusher.send_multipart((identity, status))
            except zmq.ZMQError as e:
                logger.warning(
                    "Internal error, maybe the server is shutting down. Error: %s",
                    e,
                )
            finally:
                pusher.close()

    def send_kv_to_decode(self, meta: MooncakeAgentMetadata):
        logger.info(
            "[PD] send_kv_to_decode: request_ids=%s, "
            "cp_rank=%d, cp_ws=%d, cp_interleave=%d, "
            "num_tokens=%s, remote=%s:%d",
            meta.request_ids, meta.cp_rank, meta.cp_world_size,
            meta.cp_interleave_size, meta.num_tokens,
            meta.remote_hostname, meta.remote_port,
        )
        send_reqs: list[tuple[ReqId, SendBlockMeta]] = []
        with self.reqs_need_send.lock:
            for req_id in meta.request_ids:
                send_meta = self.reqs_need_send.reqs.get(req_id)
                if send_meta is None:
                    logger.warning(
                        "[PD] Request %s not found in reqs_need_send "
                        "(available: %s)",
                        req_id, list(self.reqs_need_send.reqs.keys()),
                    )
                    return
                send_meta.expire_time = float("inf")
                if meta.cp_world_size > 1:
                    send_meta.expected_receivers = max(
                        send_meta.expected_receivers, meta.cp_world_size
                    )
                send_reqs.append((req_id, send_meta))
                logger.info(
                    "[PD] send_kv_to_decode: req=%s, "
                    "local_blocks=%d, ready=%s",
                    req_id, len(send_meta.local_block_ids),
                    send_meta.ready.is_set(),
                )

        d_tp = meta.tp_size if meta.tp_size > 0 else self.remote_tp_size
        is_cp_scatter = (
            meta.cp_world_size > 1
            or (meta.num_tokens and any(n > 0 for n in meta.num_tokens))
            or self.world_size != d_tp
        )
        if is_cp_scatter and meta.num_tokens:
            self._send_blocks_cp_scatter(send_reqs, meta)
        else:
            self._send_blocks(send_reqs, meta)

        # CP-aware: only delete after all receivers are done.
        finished_ids = []
        with self.reqs_need_send.lock:
            for req_id in meta.request_ids:
                send_meta = self.reqs_need_send.reqs.get(req_id)
                if send_meta is None:
                    continue
                send_meta.completed_receivers += 1
                if send_meta.completed_receivers >= send_meta.expected_receivers:
                    del self.reqs_need_send.reqs[req_id]
                    finished_ids.append(req_id)

        if finished_ids:
            with self.finished_sending_reqs.lock:
                self.finished_sending_reqs.set.update(finished_ids)

    def _send_blocks(
        self,
        send_reqs: list[tuple[ReqId, SendBlockMeta]],
        agent_meta: MooncakeAgentMetadata,
    ):
        src_ptrs = []
        dst_ptrs = []
        lengths = []
        local_base_addr = self.kv_caches_base_addr
        remote_base_addr = agent_meta.kv_caches_base_addr
        block_len = self.block_len
        remote_session = f"{agent_meta.remote_hostname}:{agent_meta.remote_port}"

        assert len(send_reqs) == len(agent_meta.block_ids)
        for (req_id, send_meta), remote_block_ids in zip(
            send_reqs, agent_meta.block_ids
        ):
            send_meta.ready.wait()

            num_remote_blocks = len(remote_block_ids)
            if num_remote_blocks == 0:
                continue

            local_block_ids = send_meta.local_block_ids
            # Partial prefix cache hit: just read uncomputed blocks.
            num_local_blocks = len(local_block_ids)
            assert num_local_blocks >= num_remote_blocks
            if num_local_blocks > num_remote_blocks:
                local_block_ids = local_block_ids[-num_remote_blocks:]

            # Group by indices
            group_local_block_ids, group_remote_block_ids = group_concurrent_contiguous(
                local_block_ids, remote_block_ids
            )

            for local_layer_addr, remote_layer_addr in zip(
                local_base_addr, remote_base_addr
            ):
                for group_local_block_id, group_remote_block_id in zip(
                    group_local_block_ids, group_remote_block_ids
                ):
                    src_ptrs.append(
                        local_layer_addr + group_local_block_id[0] * block_len
                    )
                    dst_ptrs.append(
                        remote_layer_addr + group_remote_block_id[0] * block_len
                    )
                    lengths.append(block_len * len(group_local_block_id))

            logger.debug(
                "Sending kv_caches for request %s (%d blocks) to %s",
                req_id,
                num_remote_blocks,
                remote_session,
            )

        start_time = time.perf_counter()
        ret_value = self.engine.batch_transfer_sync_write(
            remote_session, src_ptrs, dst_ptrs, lengths
        )
        if ret_value != 0:
            raise RuntimeError(f"Error in batch_transfer_sync_write: {ret_value}")

        logger.debug(
            "Sending to %s done, took %s",
            remote_session,
            time.perf_counter() - start_time,
        )

    def _send_blocks_cp_scatter(
        self,
        send_reqs: list[tuple[ReqId, SendBlockMeta]],
        agent_meta: MooncakeAgentMetadata,
    ):
        """CP-aware scatter: send only the positions that belong to the
        requesting decode rank, mapped from prefill's layout to decode's
        interleaved layout.

        This handles arbitrary M-to-N CP topology: prefill has
        cp_world_size=M with interleave I_p, decode has cp_world_size=N
        with interleave I_d. This prefill rank sends positions it owns
        that also belong to the target decode rank.
        """
        src_ptrs: list[int] = []
        dst_ptrs: list[int] = []
        lengths: list[int] = []
        local_base_addr = self.kv_caches_base_addr
        remote_base_addr = agent_meta.kv_caches_base_addr
        remote_session = (
            f"{agent_meta.remote_hostname}:{agent_meta.remote_port}"
        )

        # Prefill-side CP info (this rank).
        # If prefill has no domain, cp_rank=0, world_size=1.
        p_cp_rank = getattr(self, '_local_cp_rank', 0)
        p_cp_ws = getattr(self, '_local_cp_world_size', 1)
        p_interleave = getattr(
            self, '_local_cp_interleave_size',
            self.block_size,
        )
        p_block_size = self.block_size

        # Decode-side CP info (from handshake metadata).
        d_cp_rank = agent_meta.cp_rank
        d_cp_ws = agent_meta.cp_world_size
        d_interleave = agent_meta.cp_interleave_size
        d_block_size = agent_meta.cp_block_size

        # Bytes per token per layer in KV cache (local / prefill side).
        p_token_bytes = self.block_len // p_block_size

        # Remote (decode) per-token bytes and TP offset.
        # For GQA: KV heads are split by TP, so D's token size differs.
        # For MLA: KV cache is not split by TP, so same size.
        # Use D's TP from handshake (meta.tp_size) when available,
        # falling back to static config (self.remote_tp_size).
        d_tp = (
            agent_meta.tp_size
            if agent_meta.tp_size > 0
            else self.remote_tp_size
        )
        if self.use_mla or self.world_size == d_tp:
            d_token_bytes = p_token_bytes
            tp_head_offset = 0
        else:
            d_token_bytes = p_token_bytes * self.world_size // d_tp
            tp_head_offset = self.tp_rank * p_token_bytes

        logger.info(
            "[PD] cp_scatter: tp_rank=%d, p_tp=%d, d_tp=%d, "
            "p_token_bytes=%d, d_token_bytes=%d, tp_offset=%d, "
            "p_block_size=%d, d_block_size=%d, use_mla=%s",
            self.tp_rank, self.world_size, d_tp,
            p_token_bytes, d_token_bytes, tp_head_offset,
            p_block_size, d_block_size, self.use_mla,
        )

        assert len(send_reqs) == len(agent_meta.block_ids)
        assert len(send_reqs) == len(agent_meta.num_tokens)

        for (req_id, send_meta), remote_block_ids, num_tok in zip(
            send_reqs, agent_meta.block_ids, agent_meta.num_tokens
        ):
            send_meta.ready.wait()
            if num_tok == 0 or len(remote_block_ids) == 0:
                continue

            local_block_ids = send_meta.local_block_ids

            # Build position-level scatter plan using numpy.
            positions = np.arange(num_tok, dtype=np.int64)

            # Filter: positions owned by this prefill rank.
            if p_cp_ws > 1:
                p_owned = (
                    (positions // p_interleave) % p_cp_ws == p_cp_rank
                )
            else:
                p_owned = np.ones(num_tok, dtype=bool)

            # Filter: positions needed by the target decode rank.
            if d_cp_ws > 1:
                d_owned = (
                    (positions // d_interleave) % d_cp_ws == d_cp_rank
                )
            else:
                d_owned = np.ones(num_tok, dtype=bool)

            # Intersection: positions this prefill rank sends to
            # this decode rank.
            mask = p_owned & d_owned
            scatter_positions = positions[mask]

            if len(scatter_positions) == 0:
                continue

            # Compute source (prefill) block + offset.
            if p_cp_ws > 1:
                # Global position → local position on this rank.
                p_stride = p_interleave * p_cp_ws
                p_cycle_idx = scatter_positions // p_stride
                p_pos_in_chunk = scatter_positions % p_interleave
                p_local_pos = p_cycle_idx * p_interleave + p_pos_in_chunk
                p_block_idx = p_local_pos // p_block_size
                p_local_offset = p_local_pos % p_block_size
            else:
                p_block_idx = scatter_positions // p_block_size
                p_local_offset = scatter_positions % p_block_size

            # Compute dest (decode) block + offset.
            if d_cp_ws > 1:
                d_stride = d_interleave * d_cp_ws
                d_cycle_idx = scatter_positions // d_stride
                d_pos_in_chunk = scatter_positions % d_interleave
                d_local_pos = d_cycle_idx * d_interleave + d_pos_in_chunk
                d_block_idx = d_local_pos // d_block_size
                d_local_offset = d_local_pos % d_block_size
            else:
                d_block_idx = scatter_positions // d_block_size
                d_local_offset = scatter_positions % d_block_size

            # Map block indices to actual block IDs.
            p_block_idx_np = p_block_idx.astype(np.int64)
            d_block_idx_np = d_block_idx.astype(np.int64)

            # Build per-layer RDMA transfer tuples.
            for local_layer_addr, remote_layer_addr in zip(
                local_base_addr, remote_base_addr
            ):
                # Coalesce contiguous token ranges for efficiency.
                # For each position, compute src_addr and dst_addr.
                src_addrs = (
                    local_layer_addr
                    + np.array(
                        [local_block_ids[int(bi)] for bi in p_block_idx_np],
                        dtype=np.int64,
                    ) * self.block_len
                    + p_local_offset * p_token_bytes
                )
                dst_addrs = (
                    remote_layer_addr
                    + np.array(
                        [remote_block_ids[int(bi)] for bi in d_block_idx_np],
                        dtype=np.int64,
                    ) * (d_block_size * d_token_bytes)
                    + d_local_offset * d_token_bytes
                    + tp_head_offset
                )

                # Coalesce contiguous transfers.
                i = 0
                while i < len(src_addrs):
                    s, d, l = int(src_addrs[i]), int(dst_addrs[i]), p_token_bytes
                    # Merge consecutive positions if addresses are contiguous.
                    while (
                        i + 1 < len(src_addrs)
                        and int(src_addrs[i + 1]) == s + l
                        and int(dst_addrs[i + 1]) == d + l
                    ):
                        l += p_token_bytes
                        i += 1
                    src_ptrs.append(s)
                    dst_ptrs.append(d)
                    lengths.append(l)
                    i += 1

            logger.debug(
                "CP scatter for request %s: %d positions, "
                "%d RDMA ops, prefill_rank=%d decode_rank=%d",
                req_id,
                len(scatter_positions),
                len(src_ptrs),
                p_cp_rank,
                d_cp_rank,
            )

        if not src_ptrs:
            return

        # Verify source KV data is non-zero before RDMA write.
        if self.device_kv_caches and send_reqs:
            try:
                first_cache = next(iter(self.device_kv_caches.values()))
                first_req_id = send_reqs[0][0]
                first_meta = send_reqs[0][1]
                if first_meta.local_block_ids:
                    bid = first_meta.local_block_ids[0]
                    # first_cache: [2, num_blocks, block_size, ...] for
                    # split_k_and_v; index [0, bid] to get K of block bid.
                    k_blk = first_cache[0, bid].float()
                    blk_sum = k_blk.abs().sum().item()
                    blk_max = k_blk.abs().max().item()
                    nonzero = int((k_blk != 0).sum().item())
                    logger.info(
                        "[PD] KV_VERIFY src: req=%s, block=%d, "
                        "abs_sum=%.4f, abs_max=%.4f, "
                        "nonzero=%d/%d, shape=%s, "
                        "num_ops=%d, total_bytes=%d",
                        first_req_id, bid, blk_sum, blk_max,
                        nonzero, k_blk.numel(),
                        list(k_blk.shape),
                        len(src_ptrs), sum(lengths),
                    )
            except Exception as e:
                logger.warning("[PD] KV_VERIFY src failed: %s", e)

        start_time = time.perf_counter()
        ret_value = self.engine.batch_transfer_sync_write(
            remote_session, src_ptrs, dst_ptrs, lengths
        )
        if ret_value != 0:
            raise RuntimeError(
                f"Error in CP scatter batch_transfer_sync_write: "
                f"{ret_value}"
            )

        logger.debug(
            "CP scatter to %s done (%d ops), took %.3fs",
            remote_session,
            len(src_ptrs),
            time.perf_counter() - start_time,
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in mooncake."""

        logger.info("Registering KV_Caches. use_mla: %s", self.use_mla)

        kv_data_ptrs = []
        kv_data_lens = []
        seen_base_addresses = []

        split_k_and_v = self.kv_topo.split_k_and_v
        tensor_size_bytes = None
        for layer_name, cache_or_caches in kv_caches.items():
            logger.debug(
                "registering layer %s with shape %s", layer_name, cache_or_caches.shape
            )
            cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]

            for cache in cache_list:
                base_addr = cache.data_ptr()
                if base_addr in seen_base_addresses:
                    continue

                seen_base_addresses.append(base_addr)
                curr_tensor_size_bytes = cache.nbytes

                if tensor_size_bytes is None:
                    tensor_size_bytes = curr_tensor_size_bytes
                    self.num_blocks = cache.shape[0]

                assert tensor_size_bytes == curr_tensor_size_bytes, (
                    "All kv cache tensors must have the same size"
                )
                kernel_block_size = cache.shape[-2 if self.use_mla else -3]
                assert self.block_size == kernel_block_size
                kv_data_ptrs.append(base_addr)
                kv_data_lens.append(tensor_size_bytes)

        self.kv_caches_base_addr = seen_base_addresses

        ret_value = self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)
        if ret_value != 0:
            raise RuntimeError("Mooncake batch memory registration failed.")

        assert tensor_size_bytes is not None
        assert self.num_blocks != 0
        assert tensor_size_bytes % self.num_blocks == 0
        self.block_len = tensor_size_bytes // self.num_blocks
        self.device_kv_caches = kv_caches
        logger.debug(
            "registered num_blocks=%d block_len=%d", self.num_blocks, self.block_len
        )

        # No need to launch server for D node.
        if self.kv_role == "kv_consumer":
            return

        ready_event = threading.Event()
        self._mooncake_sender_t = threading.Thread(
            target=self._mooncake_sender,
            args=(
                ready_event,
                self._domain_port_base
                if self._local_cp_world_size > 1
                else self.side_channel_port,
                self.tp_rank
                + self._local_cp_rank * self.world_size,
            ),
            daemon=True,
            name="mooncake_sender",
        )
        self._mooncake_sender_t.start()
        ready_event.wait()  # Wait for listener ZMQ socket to be ready.

    async def fetch_finished_recving_reqs(self) -> set[ReqId]:
        async with self.finished_recving_reqs.lock:
            finished_recving_reqs = self.finished_recving_reqs.set
            self.finished_recving_reqs.set = set()
        return finished_recving_reqs

    def get_finished(self) -> tuple[set[str] | None, set[str] | None]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        """
        fut = None
        if self.kv_role != "kv_producer":
            fut = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_recving_reqs(), self.receiver_loop
            )

        if self.kv_role != "kv_consumer":
            with self.finished_sending_reqs.lock:
                finished_sending_reqs = self.finished_sending_reqs.set
                self.finished_sending_reqs.set = set()
        else:
            finished_sending_reqs = set()

        finished_recving_reqs = fut.result() if fut else set()

        if finished_sending_reqs or finished_recving_reqs:
            logger.info(
                "[PD] worker get_finished: tp_rank=%d finished_send=%s "
                "finished_recv=%s pending_recv=%s req_cp_sizes=%s",
                self.tp_rank,
                sorted(finished_sending_reqs),
                sorted(finished_recving_reqs),
                dict(sorted(self._recv_pending_counts.items())),
                dict(sorted(self._req_cp_sizes.items())),
            )

        # Handle timeout to avoid stranding blocks on remote.
        now = time.perf_counter()
        with self.reqs_need_send.lock:
            expired_reqs = [
                req_id
                for req_id, send_meta in self.reqs_need_send.reqs.items()
                if send_meta.expire_time < now
            ]
            for req_id in expired_reqs:
                logger.warning(
                    "Request %s timed out after %d seconds without "
                    "being sent. Freeing its blocks on the producer side.",
                    req_id,
                    envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                )
                del self.reqs_need_send.reqs[req_id]
            if expired_reqs:
                finished_sending_reqs.update(expired_reqs)

        return finished_sending_reqs or None, finished_recving_reqs or None

    async def receive_kv(
        self,
        path: str,
        req_blocks: list[tuple[str, list[int]]],
        cp_info: dict | None = None,
        local_req_ids: list[str] | None = None,
    ):
        req_ids, block_ids = map(list, zip(*req_blocks))

        # Populate CP fields from cp_info (if DyCP PD).
        cp_kwargs: dict = {}
        if cp_info:
            cp_kwargs = dict(
                cp_rank=cp_info.get("decode_cp_rank", 0),
                cp_world_size=cp_info.get("decode_cp_world_size", 1),
                cp_interleave_size=cp_info.get(
                    "decode_cp_interleave_size", 64
                ),
                cp_block_size=cp_info.get(
                    "decode_block_size", self.block_size
                ),
                num_tokens=cp_info.get("num_tokens", []),
            )

        metadata = MooncakeAgentMetadata(
            remote_hostname=self.hostname,
            remote_port=self.rpc_port,
            request_ids=req_ids,
            kv_caches_base_addr=self.kv_caches_base_addr,
            block_ids=block_ids,
            tp_size=self.world_size,
            **cp_kwargs,
        )

        encoded_data = self._encoder.encode(metadata)
        logger.info(
            "[PD] receive_kv: sending handshake to %s, "
            "remote_req_ids=%s, local_req_ids=%s, "
            "num_blocks=%s, cp_info=%s",
            path, req_ids, local_req_ids,
            [len(b) for b in block_ids],
            cp_kwargs if cp_kwargs else "none",
        )

        # Send query for the request.
        sock: zmq.asyncio.Socket = make_zmq_socket(
            self.async_zmq_ctx, path, zmq.REQ, bind=False, linger=0
        )
        sock.setsockopt(zmq.RCVTIMEO, 60000)
        try:
            await sock.send(encoded_data)
            ret_msg = await sock.recv()
            if ret_msg != TRANS_DONE:
                logger.error(
                    "Error happens during tranfering kvcache for %s, see logs in prefiller.",  # noqa: E501
                    req_ids,
                )
                return
        except zmq.ContextTerminated:
            logger.info("[PD] receive_kv: ZMQ terminated for %s", req_ids)
        except Exception as e:
            logger.error(
                "[PD] receive_kv FAILED for %s: %s", req_ids, e
            )
            return
        finally:
            sock.close()

        finished_ids = local_req_ids if local_req_ids else req_ids
        actually_finished = []
        for rid in finished_ids:
            prev_pending = self._recv_pending_counts.get(rid, 1)
            pending = prev_pending - 1
            if pending <= 0:
                actually_finished.append(rid)
                self._recv_pending_counts.pop(rid, None)
            else:
                self._recv_pending_counts[rid] = pending
            logger.info(
                "[PD] receive_kv vote: req=%s prev_pending=%d new_pending=%d finished=%s",
                rid,
                prev_pending,
                pending,
                pending <= 0,
            )

        if actually_finished:
            async with self.finished_recving_reqs.lock:
                self.finished_recving_reqs.set.update(actually_finished)

            # D-side KV verification: check blocks contain non-zero data
            # after all P TP ranks have completed their RDMA writes.
            if self.device_kv_caches:
                try:
                    import torch
                    rid_to_blks = dict(zip(finished_ids, block_ids))
                    first_cache = next(
                        iter(self.device_kv_caches.values())
                    )
                    for rid in actually_finished:
                        blk_ids = rid_to_blks.get(rid, [])
                        if not blk_ids:
                            continue
                        # Check ALL blocks, not just the first one.
                        block_checksums = []
                        head_zero_blocks = []
                        n_heads = 1
                        for bi, bid in enumerate(blk_ids):
                            k_blk = first_cache[0, bid].float()
                            v_blk = first_cache[1, bid].float()
                            k_sum = k_blk.abs().sum().item()
                            v_sum = v_blk.abs().sum().item()
                            k_nz = int((k_blk != 0).sum().item())
                            v_nz = int((v_blk != 0).sum().item())
                            n_heads = k_blk.shape[-2] if k_blk.dim() >= 2 else 1
                            head_nz = []
                            for h in range(n_heads):
                                if k_blk.dim() >= 2:
                                    hn = int(
                                        (k_blk[..., h, :] != 0).sum().item()
                                    )
                                    head_nz.append(hn)
                                    if hn == 0:
                                        head_zero_blocks.append(
                                            (bi, h)
                                        )
                                else:
                                    head_nz.append(k_nz)
                            block_checksums.append(
                                (bid, k_sum, v_sum, k_nz, v_nz, head_nz)
                            )
                        # Log first block in detail.
                        bid, k_sum, v_sum, k_nz, v_nz, head_nz = (
                            block_checksums[0]
                        )
                        k_blk0 = first_cache[0, bid].float()
                        k_max = k_blk0.abs().max().item()
                        logger.info(
                            "[PD] D_KV_VERIFY: req=%s, block=%d, "
                            "K_sum=%.2f, K_max=%.4f, K_nz=%d/%d, "
                            "V_sum=%.2f, V_nz=%d/%d, "
                            "head_nz=%s, shape=%s, "
                            "num_blocks=%d, cp_info=%s",
                            rid, bid, k_sum, k_max,
                            k_nz, k_blk0.numel(),
                            v_sum, v_nz, v_blk.numel() if 'v_blk' in dir() else 0,
                            head_nz, list(k_blk0.shape),
                            len(blk_ids),
                            cp_info if cp_info else "none",
                        )
                        # Log per-block summary for all blocks.
                        blk_summary = "; ".join(
                            f"b{bi}(id={bid}:K={ks:.0f}:V={vs:.0f}:Knz={kn}:Vnz={vn}:hnz={hn})"
                            for bi, (bid, ks, vs, kn, vn, hn) in enumerate(
                                block_checksums
                            )
                        )
                        logger.info(
                            "[PD] D_KV_VERIFY_ALL: req=%s, "
                            "num_blocks=%d, n_heads=%d, "
                            "head_zero_blocks=%s, blocks=[%s]",
                            rid, len(blk_ids), n_heads,
                            head_zero_blocks,
                            blk_summary[:2000],
                        )
                except Exception as e:
                    logger.warning("[PD] D_KV_VERIFY failed: %s", e)

            # Synchronize CUDA to ensure all RDMA writes are visible
            # to the GPU before marking the request as finished.
            # This prevents the attention from reading stale data.
            if actually_finished:
                try:
                    import torch
                    torch.cuda.synchronize()
                    logger.debug(
                        "[PD] CUDA sync after KV recv for %s",
                        actually_finished,
                    )
                except Exception as e:
                    logger.warning("[PD] CUDA sync failed: %s", e)

        logger.info(
            "[PD] receive_kv: KV pull DONE for remote=%s local=%s "
            "actually_finished=%s from %s",
            req_ids, finished_ids, actually_finished, path,
        )

    def group_kv_pull(self, metadata: MooncakeConnectorMetadata):
        """Group KV pull requests by destination path.

        When prefill has dp_per_domain > 1, each request is sent to
        each prefill DP rank separately (CP scatter). Otherwise
        standard single-path pull.

        Returns:
            dict[path, list[(remote_req_id, local_req_id, block_ids,
                             cp_info|None)]]
        """
        kv_pulls: dict[str, list] = defaultdict(list)
        for req_id, meta in metadata.reqs_to_recv.items():
            remote_req_id = meta.remote_request_id or req_id
            logger.info(
                "[PD] group_kv_pull: local_req=%s, remote_req=%s, "
                "num_blocks=%d",
                req_id, remote_req_id, len(meta.local_block_ids),
            )
            p_cp_ws = meta.prefill_cp_world_size
            p_tp_size = meta.prefill_tp_size
            need_cp_scatter = (
                p_cp_ws > 1
                or meta.decode_cp_world_size > 1
                or (p_tp_size > 1 and p_tp_size != self.world_size)
            )
            cp_info = None
            if need_cp_scatter:
                cp_info = dict(
                    decode_cp_rank=meta.decode_cp_rank,
                    decode_cp_world_size=meta.decode_cp_world_size,
                    decode_cp_interleave_size=meta.decode_cp_interleave_size,
                    decode_block_size=self.block_size,
                    num_tokens=[meta.num_tokens],
                )

            if p_cp_ws > 1:
                base_port = meta.remote_port
                dycp_ranks = (
                    meta.remote_dycp_ranks
                    if meta.remote_dycp_ranks is not None
                    else list(range(p_cp_ws))
                )
                if p_tp_size > 1 and p_tp_size != self.world_size:
                    for p_rank in dycp_ranks:
                        for p_tp_rank in range(p_tp_size):
                            port = (
                                base_port
                                + p_rank * p_tp_size
                                + p_tp_rank
                            )
                            path = make_zmq_path(
                                "tcp", meta.remote_host, port
                            )
                            kv_pulls[path].append(
                                (remote_req_id, req_id,
                                 meta.local_block_ids, cp_info)
                            )
                else:
                    for p_rank in dycp_ranks:
                        port = (
                            base_port
                            + p_rank * p_tp_size
                            + self.tp_rank
                        )
                        path = make_zmq_path(
                            "tcp", meta.remote_host, port
                        )
                        kv_pulls[path].append(
                            (remote_req_id, req_id,
                             meta.local_block_ids, cp_info)
                        )
            else:
                if p_tp_size > 1 and p_tp_size != self.world_size:
                    base_port = meta.remote_port
                    for p_tp_rank in range(p_tp_size):
                        port = base_port + p_tp_rank
                        path = make_zmq_path(
                            "tcp", meta.remote_host, port
                        )
                        kv_pulls[path].append(
                            (remote_req_id, req_id,
                             meta.local_block_ids, cp_info)
                        )
                else:
                    path = make_zmq_path(
                        "tcp", meta.remote_host,
                        meta.remote_port + self.tp_rank,
                    )
                    kv_pulls[path].append(
                        (remote_req_id, req_id,
                         meta.local_block_ids, cp_info)
                    )

        return kv_pulls

    def start_load_kv(self, metadata: MooncakeConnectorMetadata):
        if self.kv_role != "kv_producer":
            logger.info(
                "[PD] start_load_kv: reqs_to_recv=%d, reqs_to_send=%d",
                len(metadata.reqs_to_recv), len(metadata.reqs_to_send),
            )
            kv_pulls = self.group_kv_pull(metadata)

            pull_counts: dict[str, int] = defaultdict(int)
            for req_entries in kv_pulls.values():
                for _, local_rid, _, _ in req_entries:
                    pull_counts[local_rid] += 1
            for local_rid, count in pull_counts.items():
                self._recv_pending_counts[local_rid] = count
            if pull_counts:
                logger.info(
                    "[PD] start_load_kv: recv pending counts=%s",
                    dict(sorted(pull_counts.items())),
                )

            for req_id, meta in metadata.reqs_to_recv.items():
                cp_size = meta.decode_cp_world_size
                if cp_size > 1:
                    self._req_cp_sizes[req_id] = cp_size

            for path, req_entries in kv_pulls.items():
                logger.info(
                    "[PD] start_load_kv: pulling from %s, "
                    "num_reqs=%d",
                    path, len(req_entries),
                )
                # Each entry is (remote_req_id, local_req_id,
                #                block_ids, cp_info|None).
                req_blocks = [
                    (remote_rid, bids)
                    for remote_rid, _, bids, _ in req_entries
                ]
                local_req_ids = [
                    local_rid
                    for _, local_rid, _, _ in req_entries
                ]
                cp_info = req_entries[0][3] if req_entries else None
                asyncio.run_coroutine_threadsafe(
                    self.receive_kv(
                        path, req_blocks, cp_info, local_req_ids
                    ),
                    self.receiver_loop,
                )

        if self.kv_role != "kv_consumer":
            with self.reqs_need_send.lock:
                for req_id, block_ids in metadata.reqs_to_send.items():
                    logger.info(
                        "[PD] start_load_kv SEND: req=%s, "
                        "has_block_ids=%s, num_blocks=%d",
                        req_id, bool(block_ids), len(block_ids),
                    )
                    if block_ids:
                        send_meta = self.reqs_need_send.reqs.get(req_id)
                        if send_meta is None:
                            logger.warning(
                                "[PD] start_load_kv SEND: req=%s already "
                                "consumed by sender thread, skipping",
                                req_id,
                            )
                            continue
                        send_meta.local_block_ids = block_ids
                        send_meta.ready.set()
                        send_meta.expire_time = (
                            time.perf_counter()
                            + envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
                        )
                    else:
                        # From update_state_after_alloc(),
                        # but not reach request_finished() yet
                        self.reqs_need_send.reqs[req_id] = SendBlockMeta(
                            local_block_ids=[], ready=threading.Event()
                        )


def group_concurrent_contiguous(
    src_indices: list[int], dst_indices: list[int]
) -> tuple[list[list[int]], list[list[int]]]:
    """Vectorised NumPy implementation."""
    if len(src_indices) == 0:
        return [], []

    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups


def get_mooncake_side_channel_port(vllm_config: VllmConfig) -> int:
    # This logic is now centralized
    return (
        envs.VLLM_MOONCAKE_BOOTSTRAP_PORT
        + vllm_config.parallel_config.data_parallel_rank
        * vllm_config.parallel_config.tensor_parallel_size
    )
