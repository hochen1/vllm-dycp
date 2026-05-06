import os
import time
from collections import defaultdict
from collections.abc import Iterable
from typing import Any
from itertools import chain

from vllm import envs
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    EncoderDecoderCacheManager,
    compute_encoder_budget,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.cross_dp_kv_cache_manager import CrossDPKVCacheManager
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue, LongShortRequestQueue, get_cp_size_for_request
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import (
    PrefixCacheStats,
    SchedulerStats,
)
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext

from vllm.v1.core.sched.scheduler import Scheduler


logger = init_logger(__name__)

# Maximum consecutive scheduling steps where decode makes zero progress
# before force-allowing prefills. This prevents permanent deadlock when a
# decode request is stuck (e.g., NCCL hang, KV transfer stall) while
# still allowing normal decode to run without interference.
# Normal decode generates 1 token/step, so even short outputs take many
# steps. A stall count of 50 means ~50 seconds of zero progress before
# the escape hatch triggers (at ~1 step/second for decode).
_DYCP_STALL_LIMIT = 50

class RequestManager:
    def __init__(
        self,
        cp_world_size: int,
        max_num_seqs: int,
    ):
        self.cp_world_size = cp_world_size
        self.max_num_seqs = max_num_seqs
        self.num_long_req_per_domain = 0
        self.num_req_per_dp = [0] * self.cp_world_size
        # Track number of running requests per CP size for accurate
        # total request count under DyCP.
        self.num_req_per_cp_size: dict[int, int] = {}

    def select_dp(
        self,
        request: Request,
        is_long: bool,
        rank_budgets: list[int] | None = None,
        cp_size: int = 0,
    ) -> list[int] | None:
        if len(request.cp_ranks) > 0:
            if all([self.num_req_per_dp[rank] < self.max_num_seqs for rank in request.cp_ranks]):
                return request.cp_ranks
            # Stale cp_ranks from a preempted request: the old ranks
            # are occupied. Fall through to DyCP-aware rank selection
            # instead of returning None, which would block all
            # subsequent scheduling.
            if cp_size > 0:
                request.cp_ranks = []
            else:
                return None

        if cp_size > 1:
            # DyCP: select aligned rank subset of size cp_size.
            # Ranks must start at multiples of cp_size.
            best_group = None
            best_min_reqs = float('inf')
            for start in range(0, self.cp_world_size - cp_size + 1, cp_size):
                group = list(range(start, start + cp_size))
                if not all(self.num_req_per_dp[r] < self.max_num_seqs for r in group):
                    continue
                if rank_budgets is not None and not all(rank_budgets[r] > 0 for r in group):
                    continue
                max_reqs = max(self.num_req_per_dp[r] for r in group)
                if max_reqs < best_min_reqs:
                    best_min_reqs = max_reqs
                    best_group = group
            return best_group
        elif is_long and cp_size < 1:
            # Non-DyCP long request: use all ranks for context parallelism.
            # When cp_size >= 1 (DyCP enabled), the threshold logic is
            # authoritative — cp_size=1 means single rank regardless of
            # is_long. Only use the all-ranks path when DyCP is disabled
            # (cp_size == 0) and the request is classified as long.
            return [
                i for i in range(self.cp_world_size)
            ]
        else:
            if rank_budgets is not None:
                candidates = [
                    i for i in range(self.cp_world_size)
                    if self.num_req_per_dp[i] < self.max_num_seqs
                    and rank_budgets[i] > 0
                ]
                if not candidates:
                    return None
                best_dp = max(candidates, key=lambda i: rank_budgets[i])
            else:
                best_dp = min(range(len(self.num_req_per_dp)),
                              key=lambda i: self.num_req_per_dp[i])
            return [best_dp]
    
    def add_req(self, request: Request) -> None:
        if len(request.cp_ranks) > 1:
            self.num_long_req_per_domain += 1

        cp_size = len(request.cp_ranks)
        self.num_req_per_cp_size[cp_size] = \
            self.num_req_per_cp_size.get(cp_size, 0) + 1

        for rank in request.cp_ranks:
            self.num_req_per_dp[rank] += 1

    def free_req(self, request: Request) -> None:
        if len(request.cp_ranks) > 1:
            self.num_long_req_per_domain -= 1

        cp_size = len(request.cp_ranks)
        self.num_req_per_cp_size[cp_size] = \
            self.num_req_per_cp_size.get(cp_size, 0) - 1
        assert self.num_req_per_cp_size[cp_size] >= 0, (
            f"num_req_per_cp_size[{cp_size}] went negative: "
            f"{self.num_req_per_cp_size[cp_size]}"
        )
        if self.num_req_per_cp_size[cp_size] == 0:
            del self.num_req_per_cp_size[cp_size]

        for rank in request.cp_ranks:
            self.num_req_per_dp[rank] -= 1
            assert self.num_req_per_dp[rank] >= 0, (
                f"num_req_per_dp[{rank}] went negative: "
                f"{self.num_req_per_dp[rank]}"
            )

    def get_num_req_per_dp(self, dp_rank: int) -> int:
        return self.num_req_per_dp[dp_rank]

    def get_num_cp_req_per_domain(self) -> int:
        return self.num_long_req_per_domain

    def get_total_num_req(self) -> int:
        # Count unique requests by accounting for per-CP-size
        # deduplication. Each request with cp_size=N is counted N times
        # in num_req_per_dp (once per rank), so we subtract (N-1)
        # per request.
        total = sum(self.num_req_per_dp)
        for cp_size, count in self.num_req_per_cp_size.items():
            total -= count * (cp_size - 1)
        return total

    def has_slot_for_long_request(self) -> bool:
        return all(self.num_req_per_dp[i] < self.max_num_seqs for i in range(self.cp_world_size))

    def has_slot_for_cp_request(self, cp_size: int) -> bool:
        """Check if any aligned subgroup of cp_size has room for a new request.

        Unlike has_slot_for_long_request which checks ALL ranks, this method
        checks only the ranks needed for the given CP size. Under DyCP, a CP=2
        request only needs 2 consecutive ranks, not all 8.
        """
        if cp_size <= 1:
            # CP=1: any single rank with room
            return any(self.num_req_per_dp[i] < self.max_num_seqs
                       for i in range(self.cp_world_size))
        # CP>1: any aligned subgroup of cp_size with room
        for start in range(0, self.cp_world_size - cp_size + 1, cp_size):
            group = range(start, start + cp_size)
            if all(self.num_req_per_dp[r] < self.max_num_seqs for r in group):
                return True
        return False

    def __repr__(self) -> str:
        return (f"RequestManager(cp_world_size={self.cp_world_size}, "
                + f"max_num_seqs={self.max_num_seqs}, "
                + f"num_long_req_per_domain={self.num_long_req_per_domain}, "
                + f"num_req_per_dp={self.num_req_per_dp})")

class CrossDPScheduler(Scheduler):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            block_size=block_size,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )

        self.cp_world_size = vllm_config.parallel_config.dp_per_domain
        self.finished_req_ids: list[set[str]] = [set() for _ in range(self.cp_world_size)]

        self.kv_cache_manager = CrossDPKVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            cp_world_size=self.cp_world_size,
            hash_block_size=self.block_size,
            metrics_collector=self.kv_metrics_collector,
        )
        self.max_cp_tokens = self.vllm_config.scheduler_config.num_cp_seqs
        self.graph_size_for_cp = self.vllm_config.compilation_config.cudagraph_capture_sizes_for_cp
        assert self.max_cp_tokens >= self.graph_size_for_cp, "max_cp_tokens should be greater than or equal to graph_size_for_cp"

        # DyCP: threshold-based dynamic CP size
        self.dycp_enabled = vllm_config.parallel_config.dycp_enabled
        self.dycp_sorted_thresholds = (
            vllm_config.parallel_config.dycp_sorted_thresholds
            if self.dycp_enabled else []
        )

        # When DyCP is enabled, auto-align long_request_threshold with the
        # minimum CP>1 threshold so that all CP requests are classified as
        # "long" and subject to the num_cp_seqs limit. This prevents too many
        # CP requests from running concurrently and exceeding CUDA graph
        # capture sizes, which would cause fallback to eager mode.
        _env_thresh = os.environ.get("VLLM_LONG_REQUEST_THRESHOLD")
        if _env_thresh is not None:
            _thresh = int(_env_thresh)
        elif self.dycp_enabled and self.dycp_sorted_thresholds:
            _thresh = 128 * 1024  # default fallback
            for thresh, cp_size in self.dycp_sorted_thresholds:
                if cp_size > 1:
                    _thresh = thresh
                    break
            logger.info(
                "CrossDPScheduler: DyCP enabled, auto-setting "
                "long_request_threshold=%d (min CP>1 threshold)", _thresh)
        else:
            _thresh = 128 * 1024
        logger.info("CrossDPScheduler: long_request_threshold=%d", _thresh)
        self.waiting = LongShortRequestQueue(
            long_request_threshold=_thresh,
            max_long_requests=self.max_cp_tokens,
            dycp_sorted_thresholds=self.dycp_sorted_thresholds if self.dycp_enabled else None,
        )
        self.request_manager = RequestManager(
            cp_world_size=self.cp_world_size,
            max_num_seqs=self.max_num_running_reqs,
        )
        # Allow the queue to check per-CP-size slot availability under DyCP.
        if self.dycp_enabled:
            self.waiting._request_manager = self.request_manager
        # Track which requests are currently registered in request_manager
        # and counted in running_long_count. Prevents double-decrement when
        # a preempted request is later cancelled via finish_requests().
        self._active_req_ids: set[str] = set()

        # When a CP>1 request is preempted, cp_ranks is cleared so the
        # re-scheduled request goes through DyCP-aware rank selection.
        # Save the original cp_ranks here so that _free_request can still
        # notify the correct worker ranks if the request is later cancelled.
        self._preempted_cp_ranks: dict[str, list[int]] = {}

        # Escape hatch for dycp_has_decode deadlock: track whether decode
        # requests are making progress. If decode stalls for
        # _DYCP_STALL_LIMIT consecutive steps (zero new tokens computed),
        # force-allow prefills to prevent permanent deadlock.
        self._dycp_stall_count: int = 0
        self._last_decode_computed: int = 0

        # Diagnostics: track zero-progress steps to detect hangs
        self._zero_progress_steps: int = 0
        self._last_total_computed: int = 0

    def _is_long_request(self, request: Request) -> bool:
        """Classify a request as long or short, applying PD overrides.

        PD decode requests always use CP=1 (single rank). PD prefill
        requests without DyCP also use CP=1 because the decode side
        loads KV with cp_world_size=1. When DyCP is enabled, the
        threshold logic determines the correct cp_size.
        """
        is_long = self.waiting.is_long_request(request)
        kv_params = request.kv_transfer_params
        if kv_params and kv_params.get("do_remote_prefill"):
            is_long = False
        elif kv_params and kv_params.get("do_remote_decode"):
            if not self.dycp_enabled:
                is_long = False
        return is_long

    def _update_after_schedule(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            if len(request.cp_ranks) == 1:
                request.num_computed_tokens += num_scheduled_token
            elif (len(request.cp_ranks) > 1
                  and scheduler_output.cp_rank == request.cp_ranks[0]):
                # Only update on the first rank of the CP group to avoid
                # double-counting. Must use cp_ranks[0] (first rank in
                # the CP subgroup), not global rank 0 — e.g., a CP=4
                # request on ranks [4,5,6,7] has cp_ranks[0]=4.
                request.num_computed_tokens += num_scheduled_token
            elif len(request.cp_ranks) == 0:
                # Should not happen: scheduled requests always have
                # cp_ranks set. If reached, the request will hang
                # because num_computed_tokens never advances.
                logger.warning(
                    "Scheduled request %s has empty cp_ranks — "
                    "num_computed_tokens will not be updated",
                    req_id,
                )

            # NOTE: _free_encoder_inputs relies on num_computed_tokens, which
            # may be updated again in _update_from_output for speculative
            # decoding. However, it is safe to call the method here because
            # encoder inputs are always part of the prompt, not the output,
            # and thus are unaffected by speculative decoding.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

        # Clear the finished request IDs.
        # NOTE: We shouldn't do self.finished_req_ids.clear() here because
        # it will also affect the scheduler output.
        self.finished_req_ids[scheduler_output.cp_rank] = set()

    def _free_request(self, request: Request) -> dict[str, Any] | None:
        assert request.is_finished()

        # Only decrement running_long_count and free from request_manager
        # if the request is still registered as active. Preempted requests
        # are already unregistered during preemption; calling this again
        # from finish_requests() would cause double-decrement.
        if request.request_id in self._active_req_ids:
            self._active_req_ids.discard(request.request_id)
            # PD requests are classified as short (CP=1) at schedule time,
            # so they must also be classified as short at free time to keep
            # running_long_count consistent.
            if not self.dycp_enabled:
                is_long = self._is_long_request(request)
                self.waiting.running_long_count -= 1 if is_long else 0
                self.waiting.has_slot_for_long_request = \
                    self.request_manager.has_slot_for_long_request()
            self.request_manager.free_req(request)

        delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        # Use saved preempted cp_ranks if the request was preempted
        # (which clears cp_ranks). This ensures worker ranks are
        # notified that the request is finished even if it was
        # cancelled while in the preempted/waiting state.
        finished_cp_ranks = request.cp_ranks or self._preempted_cp_ranks.pop(
            request_id, [])
        for cp_rank in finished_cp_ranks:
            self.finished_req_ids[cp_rank].add(request_id)
        # Clean up saved cp_ranks if they were used
        self._preempted_cp_ranks.pop(request_id, None)

        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the KV prefix cache with DyCP-aware request cleanup.

        Overrides the base class to properly clean up DyCP state
        (request_manager counters, _active_req_ids, _preempted_cp_ranks,
        cp_ranks) when preempting running requests. Without this override,
        the base class _preempt_request does not call request_manager.free_req,
        causing counter corruption on re-scheduling.
        """
        if reset_running_requests:
            timestamp = time.monotonic()
            while self.running:
                request = self.running.pop()
                # DyCP-aware cleanup: same as the normal preemption path
                # in schedule().
                if request.request_id in self._active_req_ids:
                    self._active_req_ids.discard(request.request_id)
                    if not self.dycp_enabled:
                        is_long = self._is_long_request(request)
                        self.waiting.running_long_count -= (
                            1 if is_long else 0)
                    self.request_manager.free_req(request)
                if request.cp_ranks:
                    self._preempted_cp_ranks[request.request_id] = (
                        list(request.cp_ranks))
                    request.cp_ranks = []
                self._preempt_request(request, timestamp)
                request.num_output_placeholders = 0
                request.discard_latest_async_tokens = True
            self.prev_step_scheduled_req_ids.clear()

        reset_successful = self.kv_cache_manager.reset_prefix_cache()
        if reset_running_requests and not reset_successful:
            raise RuntimeError(
                "Failed to reset KV cache even when all the running "
                "requests are preempted and moved to the waiting queue.")
        if reset_connector and self.connector is not None:
            self.connector.reset()
        return reset_successful

    def has_finished_requests(self) -> bool:
        return sum(len(sub_ids) for sub_ids in self.finished_req_ids) > 0

    def _update_from_kv_xfer_finished(self, kv_connector_output):
        """Override to handle multi-rank deduplication.

        With multiple CP ranks, the same finished_sending req_id may
        arrive from multiple workers across scheduling steps. Guard
        against double-free by checking self.requests before freeing.
        """
        if self.connector is not None:
            self.connector.bind_connector_metadata(None)

        for req_id in kv_connector_output.finished_recving or ():
            self.finished_recving_kv_req_ids.add(req_id)
        for req_id in kv_connector_output.finished_sending or ():
            if req_id in self.requests:
                req = self.requests[req_id]
                self._free_blocks(req)
            else:
                logger.debug(
                    "Skipping finished_sending for already-freed %s",
                    req_id)

    def _connector_finished(
        self, request: Request
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None

        block_ids = self.kv_cache_manager.get_block_ids(request)

        if not isinstance(self.connector, SupportsHMA):
            # NOTE(Kuntai): We should deprecate this code path after we enforce
            # all connectors to support HMA.
            # Hybrid memory allocator should be already turned off for this
            # code path, but let's double-check here.
            assert len(self.kv_cache_config.kv_cache_groups) == 1
            return self.connector.request_finished(request, block_ids)

        return self.connector.request_finished_all_groups(request, block_ids)
    
    def update_from_output(
        self,
        scheduler_outputs: list[SchedulerOutput],
        model_runner_outputs: list[ModelRunnerOutput],
    ) -> dict[int, EngineCoreOutputs]:

        """When using example connector, many stats are None, so
        scheduler_stats only contains the last DP stats."""
        processed_request: list[str] = []
        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        
        for scheduler_output, model_runner_output in zip(scheduler_outputs, model_runner_outputs):
            if model_runner_output is False:
                continue

            sampled_token_ids = model_runner_output.sampled_token_ids
            logprobs = model_runner_output.logprobs
            prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens
            pooler_outputs = model_runner_output.pooler_output
            num_nans_in_logits = model_runner_output.num_nans_in_logits
            kv_connector_output = model_runner_output.kv_connector_output
            cudagraph_stats = model_runner_output.cudagraph_stats

            # outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
            spec_decoding_stats: SpecDecodingStats | None = None
            kv_connector_stats: KVConnectorStats | None = (
                kv_connector_output.kv_connector_stats if kv_connector_output else None
            )
            if kv_connector_stats and self.connector:
                kv_stats = self.connector.get_kv_connector_stats()
                assert kv_stats is None, (
                    "kv_stats must be None for the example connector; "
                    "implement aggregation if needed")
                if kv_stats:
                    kv_connector_stats = kv_connector_stats.aggregate(kv_stats)

            failed_kv_load_req_ids = None
            if kv_connector_output and kv_connector_output.invalid_block_ids:
                # These blocks contain externally computed tokens that failed to
                # load. Identify affected requests and adjust their computed token
                # count to trigger recomputation of the invalid blocks.
                failed_kv_load_req_ids = self._handle_invalid_blocks(
                    kv_connector_output.invalid_block_ids
                )

            # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
            # the below loop can be a performance bottleneck. We should do our best
            # to avoid expensive operations inside the loop.
            stopped_running_reqs: set[Request] = set()
            stopped_preempted_reqs: set[Request] = set()
            for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
                """
                Remove the duplicate model output
                """
                if req_id in processed_request:
                    continue

                assert num_tokens_scheduled > 0
                if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                    # skip failed or rescheduled requests from KV load failure
                    continue
                request = self.requests.get(req_id)
                if request is None:
                    # The request is already finished. This can happen if the
                    # request is aborted while the model is executing it (e.g.,
                    # in pipeline parallelism).
                    continue

                req_index = model_runner_output.req_id_to_index[req_id]
                generated_token_ids = (
                    sampled_token_ids[req_index] if sampled_token_ids else []
                )

                scheduled_spec_token_ids = (
                    scheduler_output.scheduled_spec_decode_tokens.get(req_id)
                )
                if scheduled_spec_token_ids:
                    num_draft_tokens = len(scheduled_spec_token_ids)
                    num_accepted = len(generated_token_ids) - 1
                    num_rejected = num_draft_tokens - num_accepted
                    # num_computed_tokens represents the number of tokens
                    # processed in the current step, considering scheduled
                    # tokens and rejections. If some tokens are rejected,
                    # num_computed_tokens is decreased by the number of rejected
                    # tokens.
                    if request.num_computed_tokens > 0:
                        request.num_computed_tokens -= num_rejected
                    # If async scheduling, num_output_placeholders also includes
                    # the scheduled spec tokens count and so is similarly adjusted.
                    if request.num_output_placeholders > 0:
                        request.num_output_placeholders -= num_rejected
                    spec_decoding_stats = self.make_spec_decoding_stats(
                        spec_decoding_stats,
                        num_draft_tokens=num_draft_tokens,
                        num_accepted_tokens=num_accepted,
                    )

                stopped = False
                new_logprobs = None
                new_token_ids = generated_token_ids
                pooler_output = pooler_outputs[req_index] if pooler_outputs else None
                kv_transfer_params = None
                status_before_stop = request.status

                # Check for stop and update request status.
                if new_token_ids:
                    new_token_ids, stopped = self._update_request_with_output(
                        request, new_token_ids
                    )
                elif request.pooling_params and pooler_output is not None:
                    # Pooling stops as soon as there is output.
                    request.status = RequestStatus.FINISHED_STOPPED
                    stopped = True

                if stopped:
                    kv_transfer_params = self._free_request(request)
                    if status_before_stop == RequestStatus.RUNNING:
                        stopped_running_reqs.add(request)
                    else:
                        stopped_preempted_reqs.add(request)

                # Extract sample logprobs if needed.
                if (
                    request.sampling_params is not None
                    and request.sampling_params.logprobs is not None
                    and logprobs
                ):
                    new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

                if new_token_ids and self.structured_output_manager.should_advance(request):
                    struct_output_request = request.structured_output_request
                    assert struct_output_request is not None
                    assert struct_output_request.grammar is not None
                    struct_output_request.grammar.accept_tokens(req_id, new_token_ids)

                if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                    request.num_nans_in_logits = num_nans_in_logits[req_id]

                # Get prompt logprobs for this request.
                prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
                if new_token_ids or pooler_output is not None or kv_transfer_params:
                    # Add EngineCoreOutput for this Request.
                    outputs[request.client_index].append(
                        EngineCoreOutput(
                            request_id=req_id,
                            new_token_ids=new_token_ids,
                            finish_reason=request.get_finished_reason(),
                            new_logprobs=new_logprobs,
                            new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                            pooling_output=pooler_output,
                            stop_reason=request.stop_reason,
                            events=request.take_events(),
                            kv_transfer_params=kv_transfer_params,
                            trace_headers=request.trace_headers,
                            num_cached_tokens=request.num_cached_tokens,
                            num_nans_in_logits=request.num_nans_in_logits,
                        )
                    )
                else:
                    # Invariant: EngineCore returns no partial prefill outputs.
                    assert not prompt_logprobs_tensors

                processed_request.append(req_id)

            # Remove the stopped requests from the running and waiting queues.
            if stopped_running_reqs:
                self.running = remove_all(self.running, stopped_running_reqs)
            if stopped_preempted_reqs:
                # This is a rare case and unlikely to impact performance.
                self.waiting.remove_requests(stopped_preempted_reqs)

            if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
                requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
                self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
                for request in requests:
                    outputs[request.client_index].append(
                        EngineCoreOutput(
                            request_id=request.request_id,
                            new_token_ids=[],
                            finish_reason=request.get_finished_reason(),
                            events=request.take_events(),
                            trace_headers=request.trace_headers,
                            num_cached_tokens=request.num_cached_tokens,
                        )
                    )

            # KV Connector: update state for finished KV Transfers.
            if kv_connector_output:
                self._update_from_kv_xfer_finished(kv_connector_output)

            # collect KV cache events from KV cache manager
            events = self.kv_cache_manager.take_events()

            # collect KV cache events from connector
            if self.connector is not None:
                connector_events = self.connector.take_events()
                if connector_events:
                    if events is None:
                        events = list(connector_events)
                    else:
                        events.extend(connector_events)

            # publish collected KV cache events
            if events:
                batch = KVEventBatch(ts=time.time(), events=events)
                self.kv_event_publisher.publish(batch)

            # Create EngineCoreOutputs for all clients that have requests with
            # outputs in this step.
            engine_core_outputs = {
                client_index: EngineCoreOutputs(outputs=outs)
                for client_index, outs in outputs.items()
            }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set
                    )
            finished_req_ids.clear()
                
        if (
            stats := self.make_stats(
                spec_decoding_stats, kv_connector_stats, cudagraph_stats
            )
        ) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats
        
        return engine_core_outputs
    
    def schedule(self) -> list[SchedulerOutput]:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[list[Request]] = [[] for _ in range(self.cp_world_size)]
        scheduled_resumed_reqs: list[list[Request]] = [[] for _ in range(self.cp_world_size)]
        scheduled_running_reqs: list[list[Request]] = [[] for _ in range(self.cp_world_size)]
        preempted_reqs: list[list[Request]] = [[] for _ in range(self.cp_world_size)]

        req_to_new_blocks: list[dict[str, KVCacheBlocks]] = [{} for _ in range(self.cp_world_size)]
        num_scheduled_tokens: list[dict[str, int]] = [{} for _ in range(self.cp_world_size)]
        cp_rank_scheduled_tokens: list[dict[str, int]] = [{} for _ in range(self.cp_world_size)]

        # DyCP: track per-request cp_size for the output.
        # Pre-populate from ALL running requests so that actual_cp_size
        # is correct even if some running requests are skipped in the
        # scheduling loop (e.g. num_new_tokens == 0 due to budget
        # exhaustion or async scheduling).
        per_req_cp_sizes: dict[str, int] = {}
        if self.dycp_enabled:
            for req in self.running:
                if len(req.cp_ranks) > 1:
                    per_req_cp_sizes[req.request_id] = len(req.cp_ranks)

        # DyCP: enforce single CP>1 size per batch. The NCCL all-gather/
        # all-reduce in both prefill and decode paths uses a single
        # actual_cp_size for the entire batch. Mixing different CP>1 sizes
        # would cause the wrong NCCL subgroup to be used for some requests.
        # This constraint matches the design doc: "优先实现一个batch里面
        # 只会有一个size的CP".
        # Initialize from ALL RUNNING CP>1 requests (both prefill and decode).
        # Decode must be included because the NCCL all-reduce in the decode
        # path uses actual_cp_size for subgroup selection. If CP=4 decode
        # and CP=2 decode run in the same batch, actual_cp_size=max(4,2)=4,
        # causing CP=2 ranks to use get_dycp_subgroup(4) which includes
        # idle ranks that don't participate in the all-reduce → deadlock.
        dycp_batch_cp_size: int = 0
        if self.dycp_enabled:
            running_cp_sizes = {
                len(req.cp_ranks) for req in self.running
                if len(req.cp_ranks) > 1
            }
            if running_cp_sizes:
                dycp_batch_cp_size = max(running_cp_sizes)

        # Per-rank token budgets: each rank can process up to
        # max_num_scheduled_tokens.  CP requests split tokens across ranks,
        # so their per-rank cost is num_tokens / cp_size.
        rank_budgets = [self.max_num_scheduled_tokens] * self.cp_world_size

        def _get_effective_budget(cp_ranks: list[int]) -> int:
            """Return the max tokens a request on *cp_ranks* can schedule.

            For CP requests, the budget must account for DualChunkSwap
            alignment in PCP: tokens are padded to a multiple of
            ``2 * cp_size`` then divided by ``cp_size``, giving a per-rank
            cost of ``ceil(T / (2 * W)) * 2``.  The inverse is
            ``(min_budget // 2) * 2 * W``.  We still allow at least 1 token
            (decode) when the budget is positive.
            """
            cp_size = len(cp_ranks)
            if cp_size > 1:
                min_budget = min(rank_budgets[r] for r in cp_ranks)
                if min_budget <= 0:
                    return 0
                # Max total tokens whose DualChunkSwap per-rank cost fits
                # within min_budget; guarantee at least 1 for decode.
                return max((min_budget // 2) * 2 * cp_size, 1)
            return rank_budgets[cp_ranks[0]]

        def _deduct_budget(cp_ranks: list[int], num_tokens: int) -> None:
            """Deduct per-rank cost from *rank_budgets* in-place.

            For CP prefill (cp_size > 1, num_tokens > 1), use the same
            DualChunkSwap-aligned formula as
            PCPManager.update_tokens_for_pcp():
            ``per_rank_cost = ceil(num_tokens / (2 * cp_size)) * 2``.

            For CP decode (num_tokens == 1) the token is duplicated across
            ranks, so each rank processes exactly 1 token.
            """
            cp_size = len(cp_ranks)
            if cp_size > 1 and num_tokens > 1:
                per_rank_cost = (
                    (num_tokens + 2 * cp_size - 1) // (2 * cp_size)) * 2
            else:
                per_rank_cost = num_tokens
            for r in cp_ranks:
                rank_budgets[r] -= per_rank_cost

        # # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()


        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and max(rank_budgets) > 0:
            request = self.running[req_index]
                
            if (
                request.num_output_placeholders > 0
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
                # Since output placeholders are also included in the computed tokens
                # count, we subtract (num_output_placeholders - 1) to remove any draft
                # tokens, so that we can be sure no further steps are needed even if
                # they are all rejected.
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # Async scheduling: Avoid scheduling an extra step when we are sure that
                # the previous step has reached request.max_tokens. We don't schedule
                # partial draft tokens since this prevents uniform decode optimizations.
                req_index += 1
                continue

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )

            """
            TODO(AoChen): Long prefill token threshold is not implemented yet. We temporarily ignore this for decode instance.
            """
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            eff_budget = _get_effective_budget(request.cp_ranks)
            num_new_tokens = min(num_new_tokens, eff_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens
            )

            """
            TODO(AoChen): Encoder inputs scheduling is not implemented yet.
            """

            if num_new_tokens == 0:
                # Warn for prefill requests stuck at computed=0
                # (critical hang indicator).
                kv_params = request.kv_transfer_params
                is_pd_decode = (
                    kv_params and kv_params.get("do_remote_prefill"))
                if (request.num_computed_tokens == 0
                        and request.num_prompt_tokens > 0
                        and not is_pd_decode):
                    logger.warning(
                        "Prefill req=%s stuck at computed=0, "
                        "num_prompt=%d cp_ranks=%s running=%d",
                        request.request_id,
                        request.num_prompt_tokens,
                        request.cp_ranks,
                        len(self.running),
                    )
                req_index += 1
                continue

            # Schedule newly needed KV blocks for the request.
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request.cp_ranks,
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )
                    logger.debug("new_blocks: %s, request.cp_ranks: %s, num_new_tokens: %s",
                                 new_blocks, request.cp_ranks, num_new_tokens)
                    if new_blocks is not None:
                        # The request can be scheduled.
                        break
                    
                    """
                    TODO(AoChen): PRIORITY is not implemented yet.
                    """
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        raise NotImplementedError
                    else:
                        preempted_req = self.running.pop()
                        if len(preempted_req.cp_ranks) > 1:
                            # Cannot preempt CP>1 requests (would leave
                            # other ranks in the CP group in an
                            # inconsistent state).  Put it back and
                            # search for a CP=1 request instead.
                            self.running.append(preempted_req)
                            preempted_req = None
                            # Search backward (lowest priority first)
                            # for a preemptible CP=1 request.
                            for i in range(len(self.running) - 1,
                                           -1, -1):
                                if len(self.running[i].cp_ranks) <= 1:
                                    preempted_req = self.running.pop(i)
                                    break
                            if preempted_req is None:
                                logger.warning(
                                    "Cannot preempt: all running "
                                    "requests are CP>1. Waiting "
                                    "for one to finish.")
                                break
                        self._active_req_ids.discard(preempted_req.request_id)
                        self.request_manager.free_req(preempted_req)
                        if not self.dycp_enabled:
                            _is_long = self._is_long_request(preempted_req)
                            self.waiting.running_long_count -= 1 if _is_long else 0
                            self.waiting.has_slot_for_long_request = self.request_manager.has_slot_for_long_request()

                    self._preempt_request(preempted_req, scheduled_timestamp)

                    for rank in preempted_req.cp_ranks:
                        preempted_reqs[rank].append(preempted_req)
                    # Save original cp_ranks before clearing so that
                    # _free_request can notify the correct worker ranks
                    # if the request is later cancelled while preempted.
                    if preempted_req.cp_ranks:
                        self._preempted_cp_ranks[preempted_req.request_id] = (
                            list(preempted_req.cp_ranks))
                    # Clear cp_ranks so the re-scheduled request goes
                    # through DyCP-aware rank selection instead of
                    # trying stale ranks that may now be occupied.
                    preempted_req.cp_ranks = []

                    # preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        break

            if new_blocks is None:
                # Cannot schedule this request.
                break
            
            assert len(request.cp_ranks) == len(new_blocks)
            # Schedule the request.
            for i, rank in enumerate(request.cp_ranks):
                scheduled_running_reqs[rank].append(request)
                req_to_new_blocks[rank][request.request_id] = new_blocks[i]
                num_scheduled_tokens[rank][request.request_id] = num_new_tokens
                cp_rank_scheduled_tokens[rank][request.request_id] = len(request.cp_ranks)

            # DyCP: record cp_size for running requests
            if self.dycp_enabled:
                per_req_cp_sizes[request.request_id] = len(request.cp_ranks)

            _deduct_budget(request.cp_ranks, num_new_tokens)
            req_index += 1

        # DyCP: when CP>1 decode is running, defer new prefills to avoid
        # mixing prefill and CP>1 decode across DP ranks. The MoE all-to-all
        # forces all ranks to synchronize, so decode ranks would wait for
        # slower prefill ranks, degrading TPOT from ~7ms to ~80ms.
        # When only CP=1 decode is running, prefills are not deferred because
        # CP=1 ranks are independent DP ranks — the MoE sync behavior matches
        # the non-DyCP DP baseline, and mixing is safe (same as baseline).
        # ESCAPE HATCH: if decode makes zero progress for
        # _DYCP_STALL_LIMIT consecutive steps (stuck due to NCCL hang,
        # KV transfer stall, etc.), force-allow prefills to prevent
        # permanent deadlock. We track total computed tokens of decode
        # requests to distinguish normal progress from a true stall.
        dycp_has_decode = (
            self.dycp_enabled
            and any(
                req.num_computed_tokens >= req.num_prompt_tokens
                for req in self.running
            )
        )
        if dycp_has_decode:
            total_decode_computed = sum(
                req.num_computed_tokens
                for req in self.running
                if req.num_computed_tokens >= req.num_prompt_tokens
            )
            if total_decode_computed == self._last_decode_computed:
                self._dycp_stall_count += 1
            else:
                self._dycp_stall_count = 0
            self._last_decode_computed = total_decode_computed
        else:
            self._dycp_stall_count = 0
            self._last_decode_computed = 0
        # Only defer prefills when CP>1 decode is running. CP=1-only decode
        # does not need deferral — its MoE all-to-all behavior is identical
        # to the non-DyCP DP baseline where prefill/decode mixing is allowed.
        dycp_has_cp_decode = (
            self.dycp_enabled
            and any(
                len(req.cp_ranks) > 1
                and req.num_computed_tokens >= req.num_prompt_tokens
                for req in self.running
            )
        )
        dycp_defer_prefills = (
            dycp_has_cp_decode
            and self._dycp_stall_count < _DYCP_STALL_LIMIT
        )
        # Check if any running request is a CP>1 prefill (still computing
        # prompt tokens). PD decode requests must wait for these to finish
        # to avoid being forced into a CP>1 batch.
        dycp_has_cp_prefill = (
            self.dycp_enabled
            and any(
                len(req.cp_ranks) > 1
                and req.num_computed_tokens < req.num_prompt_tokens
                for req in self.running
            )
        )
        if dycp_has_decode and self._dycp_stall_count >= _DYCP_STALL_LIMIT:
            logger.warning(
                "DyCP decode stall detected (%d steps with zero progress): "
                "force-allowing prefills despite active decode. "
                "Decode may be stuck (NCCL hang or KV transfer stall).",
                self._dycp_stall_count)

        # Zero-progress hang detection
        total_computed = sum(
            req.num_computed_tokens for req in self.running)
        if total_computed == self._last_total_computed and self.running:
            self._zero_progress_steps += 1
        else:
            self._zero_progress_steps = 0
        self._last_total_computed = total_computed
        if self._zero_progress_steps > 0 and self._zero_progress_steps % 100 == 0:
            logger.warning(
                "Zero progress for %d steps: "
                "has_decode=%s has_cp_decode=%s has_cp_prefill=%s "
                "stall_count=%d running=%d waiting=%d",
                self._zero_progress_steps,
                dycp_has_decode, dycp_has_cp_decode,
                dycp_has_cp_prefill,
                self._dycp_stall_count,
                len(self.running), len(self.waiting))

        # Use a temporary RequestQueue to collect requests that need to be
        # skipped and put back at the head of the waiting queue later
        skipped_waiting_requests = create_request_queue(self.policy)

        # Next, schedule the WAITING requests.
        if not any(preempted_reqs):
            while self.waiting and max(rank_budgets) > 0:
                # Capacity check: break if no rank has room for any request.
                # Under DyCP, the old formula (assuming each long request
                # occupies all cp_world_size ranks) is too conservative.
                # Use per-rank counts from request_manager instead.
                if all(self.request_manager.num_req_per_dp[i] >= self.request_manager.max_num_seqs
                       for i in range(self.cp_world_size)):
                    break
                request = self.waiting.peek_request()
                if request is None:
                    break

                is_long = self._is_long_request(request)

                kv_params = request.kv_transfer_params

                # DyCP: determine cp_size from thresholds
                req_cp_size = 1
                if self.dycp_enabled:
                    num_prompt_tokens = request.num_tokens - request.num_output_tokens
                    req_cp_size = get_cp_size_for_request(
                        num_prompt_tokens, self.dycp_sorted_thresholds
                    )
                    # PD decode requests always run on a single rank (CP=1).
                    # KV is loaded via IPC from prefill ranks; no CP
                    # communication is needed during decode.
                    if kv_params and kv_params.get("do_remote_prefill"):
                        req_cp_size = 1
                    # When CP>1 decode is running, defer new prefill requests
                    # to avoid MoE all-to-all sync bottleneck across
                    # mixed-phase DP ranks.
                    # PD decode requests (do_remote_prefill) load KV via IPC
                    # memory copy, not a full prefill forward pass, so they
                    # don't cause MoE sync issues and should not be deferred.
                    # After _DYCP_DEFER_LIMIT consecutive deferrals, force-
                    # allow prefills to prevent deadlock when decode is stuck.
                    if (dycp_defer_prefills and num_prompt_tokens > 0
                            and not (kv_params
                                     and kv_params.get("do_remote_prefill"))):
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue
                    # When CP>1 prefill is running, defer PD decode requests.
                    # PD decode runs with CP=1 (KV loaded to a single rank),
                    # but mixing it in a CP>1 batch forces wrong actual_cp_size.
                    if (kv_params and kv_params.get("do_remote_prefill")
                            and dycp_has_cp_prefill):
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue
                    # Enforce single CP>1 size per batch: if a CP>1
                    # request was already scheduled with a different
                    # cp_size, defer this request to avoid mixing CP
                    # sizes in the same NCCL all-gather/all-reduce.
                    if (req_cp_size > 1
                            and dycp_batch_cp_size > 0
                            and req_cp_size != dycp_batch_cp_size):
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                selected_dp = self.request_manager.select_dp(
                    request, is_long,
                    rank_budgets=rank_budgets,
                    cp_size=req_cp_size if self.dycp_enabled else 0,
                )
                if selected_dp is None:
                    # No aligned subgroup available for this CP>1
                    # request. Skip it and try the next request (which
                    # may be CP=1 and fit on an available rank) rather
                    # than breaking the entire waiting loop.
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue
                
                if len(selected_dp) > 1:
                    logger.debug("CP req: selected_dp=%s, request_id=%s", selected_dp, request.request_id)
                else:
                    logger.debug("Short req: selected_dp=%s, request_id=%s", selected_dp, request.request_id)

                # KVTransfer: skip request if still waiting for remote kvs.
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    is_ready = self._update_waiting_for_remote_kv(request)
                    if is_ready:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Skip request if the structured output request is still waiting
                # for FSM compilation.

                num_external_computed_tokens = 0
                load_kv_async = False

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    # Get locally-cached tokens.
                    new_computed_blocks, num_new_local_computed_tokens = (
                        self.kv_cache_manager.get_computed_blocks(request)
                    )
                    
                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens
                            )
                        )

                        if ext_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            continue

                        request.num_external_computed_tokens = ext_tokens
                        num_external_computed_tokens = ext_tokens

                    # Total computed tokens (local + external).
                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                else:
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed.
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                if load_kv_async:
                    # KVTransfer: loading remote KV, do not allocate for new work.
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                else:
                    # Number of tokens to be scheduled.
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    effective_budget = _get_effective_budget(selected_dp)
                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > effective_budget
                    ):
                        if len(selected_dp) > 1:
                            # CP request too large — skip it and keep
                            # looking for shorter DP requests.
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            continue
                        # DP request cannot fit on this rank.
                        break

                    num_new_tokens = min(num_new_tokens, effective_budget)
                    if num_new_tokens <= 0:
                        if len(selected_dp) > 1:
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            continue
                        break

                # Handles an edge case when P/D Disaggregation
                # is used with Spec Decoding where an
                # extra block gets allocated which
                # creates a mismatch between the number
                # of local and remote blocks.
                effective_lookahead_tokens = (
                    0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens
                )

                # Determine if we need to allocate cross-attention blocks.
                if self.is_encoder_decoder and request.has_encoder_inputs:
                    # TODO(russellb): For Whisper, we know that the input is
                    # always padded to the maximum length. If we support other
                    # encoder-decoder models, this will need to be updated if we
                    # want to only allocate what is needed.
                    num_encoder_tokens = (
                        self.scheduler_config.max_num_encoder_input_tokens
                    )
                else:
                    num_encoder_tokens = 0

                new_blocks = self.kv_cache_manager.allocate_slots(
                    selected_dp,
                    request,
                    num_new_tokens + num_external_computed_tokens,
                    num_new_local_computed_tokens,
                    new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=num_encoder_tokens,
                )
                logger.debug("new_blocks: %s, request.cp_ranks: %s, num_new_tokens: %s",
                             new_blocks, request.cp_ranks, num_new_tokens)
                if new_blocks is None:
                    # The request cannot be scheduled.
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                request.cp_ranks = selected_dp
                if self.dycp_enabled:
                    per_req_cp_sizes[request.request_id] = req_cp_size
                    if req_cp_size > 1 and dycp_batch_cp_size == 0:
                        dycp_batch_cp_size = req_cp_size

                """
                TODO(AoChen): update_state_after_alloc(PD disagg) is not implemented yet.
                """
                if self.connector is not None:
                    """
                        In the example connector, new_computed_blocks + new_blocks is not used,
                        so temporarily ignore it.
                    """
                    self.connector.update_state_after_alloc(
                        request=request,
                        # new_computed_blocks + new_blocks,
                        blocks=None,
                        num_external_tokens=num_external_computed_tokens,
                    )

                # Request was already popped from self.waiting
                # unless it was re-added above due to new_blocks being None.
                request = self.waiting.pop_request()
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    skipped_waiting_requests.prepend_request(request)
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    continue

                self._update_connector_prefix_cache_stats(request)
                
                self.running.append(request)
                self._active_req_ids.add(request.request_id)
                # Clean up saved preempted cp_ranks since the request
                # has been re-scheduled with new cp_ranks.
                self._preempted_cp_ranks.pop(request.request_id, None)
                if not self.dycp_enabled:
                    self.waiting.running_long_count += 1 if is_long else 0
                    self.waiting.has_slot_for_long_request = self.request_manager.has_slot_for_long_request()
                self.request_manager.add_req(request)

                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                
                blocks = self.kv_cache_manager.get_blocks(request)

                for idx, rank in enumerate(request.cp_ranks):
                    if request.status == RequestStatus.WAITING:
                        scheduled_new_reqs[rank].append(request)
                    elif request.status == RequestStatus.PREEMPTED:
                        scheduled_resumed_reqs[rank].append(request)
                    else:
                        raise RuntimeError(f"Invalid request status: {request.status}")
                    
                    req_to_new_blocks[rank][request.request_id] = blocks[idx]

                    num_scheduled_tokens[rank][request.request_id] = num_new_tokens
                    cp_rank_scheduled_tokens[rank][request.request_id] = len(request.cp_ranks)
                    
                _deduct_budget(request.cp_ranks, num_new_tokens)
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # Count the number of prefix cached tokens.
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                
        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

        # Check if the scheduling constraints are satisfied.
        # Verify per-rank effective tokens don't exceed the per-rank limit.
        for idx in range(self.cp_world_size):
            effective_rank_tokens = 0
            for req_id, tokens in num_scheduled_tokens[idx].items():
                cp_size = cp_rank_scheduled_tokens[idx].get(req_id, 1)
                if cp_size > 1 and tokens > 1:
                    # DualChunkSwap alignment (must match _deduct_budget).
                    effective_rank_tokens += (
                        (tokens + 2 * cp_size - 1) // (2 * cp_size)) * 2
                else:
                    effective_rank_tokens += tokens
            assert effective_rank_tokens <= self.max_num_scheduled_tokens, (
                f"rank {idx} effective tokens {effective_rank_tokens} "
                f"> {self.max_num_scheduled_tokens}"
            )

        assert all(b >= 0 for b in rank_budgets), (
            f"rank_budgets underflow: {rank_budgets}"
        )
        # Under DyCP, a CP>1 request only occupies cp_size ranks (not all
        # cp_world_size), so the total unique request count is bounded by
        # max_num_seqs * cp_world_size (all CP=1 scenario).
        assert len(self.running) <= self.max_num_running_reqs * self.cp_world_size
        
        total_scheduled = (
            len(list(chain.from_iterable(scheduled_new_reqs)))
            + len(list(chain.from_iterable(scheduled_resumed_reqs)))
            + len(list(chain.from_iterable(scheduled_running_reqs)))
        )
        assert total_scheduled <= self.max_num_running_reqs * self.cp_world_size

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)

        assert sum(len(sub) for sub in scheduled_resumed_reqs) == 0, "Scheduled resumed requests are not supported now."
        
        # Construct the scheduler output.
        if self.use_v2_model_runner:
            raise NotImplementedError
        else:
            total_new_reqs_data = [
                [
                    NewRequestData.from_request(
                        req, req_to_new_blocks[idx][req.request_id].get_block_ids()
                    )
                    for req in scheduled_new_reqs[idx]
                ] for idx in range(self.cp_world_size)
            ]
        
        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            total_cached_reqs_data = [
                    self._make_cached_request_data(
                        scheduled_running_reqs[idx],
                        scheduled_resumed_reqs[idx],
                        num_scheduled_tokens[idx], # num_scheduled_tokens should be refined num_scheduled_tokens[idx]
                        scheduled_spec_decode_tokens,
                        req_to_new_blocks[idx], # req_to_new_blocks should be refined req_to_new_blocks[idx]
                    ) for idx in range(self.cp_world_size)
            ]

        # Record the request ids that were scheduled in this step.
        self.prev_step_scheduled_req_ids.clear()
        for scheduled_tokens in num_scheduled_tokens:
            self.prev_step_scheduled_req_ids.update(scheduled_tokens.keys())

        total_scheduler_output = []

        none_tokens_in_peer_sched = all([sum(num_scheduled_tokens[idx].values()) == 0 for idx in range(self.cp_world_size)])

        # DyCP: compute batch-level actual_cp_size (max across all scheduled reqs)
        # Only meaningful when DyCP is enabled; non-DyCP batches use cp_size=1.
        if self.dycp_enabled and per_req_cp_sizes:
            actual_cp_size = max(per_req_cp_sizes.values())
        else:
            actual_cp_size = 1
        if self.dycp_enabled and actual_cp_size > 1:
            logger.debug("DyCP: actual_cp_size=%d, per_req_cp_sizes=%s",
                         actual_cp_size,
                         {k: v for k, v in per_req_cp_sizes.items() if v > 1})

        for idx in range(self.cp_world_size):

            if sum(num_scheduled_tokens[idx].values()) == 0 and len(preempted_reqs[idx]) == 0 and len(self.finished_req_ids[idx]) == 0:
                scheduler_output = SchedulerOutput.make_empty()
                scheduler_output.none_tokens_in_peer_sched = none_tokens_in_peer_sched
                total_scheduler_output.append(scheduler_output)
            else:
                total_scheduler_output.append(
                    SchedulerOutput(
                        scheduled_new_reqs=total_new_reqs_data[idx],
                        scheduled_cached_reqs=total_cached_reqs_data[idx],
                        num_scheduled_tokens=num_scheduled_tokens[idx], # num_scheduled_tokens should be refined
                        total_num_scheduled_tokens=sum(num_scheduled_tokens[idx].values()), # num_scheduled_tokens should be refined
                        scheduled_spec_decode_tokens=scheduled_spec_decode_tokens, # num_scheduled_tokens should be refined
                        scheduled_encoder_inputs=scheduled_encoder_inputs, # num_scheduled_tokens should be refined
                        num_common_prefix_blocks=num_common_prefix_blocks, # num_scheduled_tokens should be refined
                        preempted_req_ids={req.request_id for req in preempted_reqs[idx]},
                        # finished_req_ids is an existing state in the scheduler,
                        # instead of being newly scheduled in this step.
                        # It contains the request IDs that are finished in between
                        # the previous and the current steps.
                        finished_req_ids=self.finished_req_ids[idx],
                        free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
                        cp_rank=idx,
                        cp_rank_scheduled_tokens=cp_rank_scheduled_tokens[idx],
                        num_cp_request=sum([1 if cp_size > 1 else 0 for cp_size in  cp_rank_scheduled_tokens[idx].values()]),
                        actual_cp_size=actual_cp_size,
                        per_req_cp_sizes=per_req_cp_sizes if self.dycp_enabled else None,
                        none_tokens_in_peer_sched=none_tokens_in_peer_sched
                    )
                )
        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            for scheduler_output in total_scheduler_output:
                if scheduler_output is None:
                    continue
                meta: KVConnectorMetadata = self.connector.build_connector_meta(
                    scheduler_output
                )
                scheduler_output.kv_connector_metadata = meta

            # Free blocks for orphaned prefill requests whose decode
            # partner never arrived (timed out via orphan cleanup in
            # build_connector_meta). Without this, blocks leak forever.
            if hasattr(self.connector, '_orphaned_prefill_ids_to_free'):
                for req_id in self.connector._orphaned_prefill_ids_to_free:
                    if req_id in self.requests:
                        logger.info(
                            "Freeing orphaned prefill blocks for %s", req_id)
                        self._free_blocks(self.requests[req_id])
                    else:
                        logger.debug(
                            "Skipping orphaned prefill free for %s "
                            "(already freed)", req_id)
                self.connector._orphaned_prefill_ids_to_free.clear()

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            for scheduler_output in total_scheduler_output:
                if scheduler_output is None:
                    continue
                self._update_after_schedule(scheduler_output)
        
        return total_scheduler_output


class AsyncCrossDPScheduler(CrossDPScheduler):
    def _update_after_schedule(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        super()._update_after_schedule(scheduler_output)
        pending_structured_output_tokens = False
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            if (
                request.num_computed_tokens
                == request.num_tokens
                + request.num_output_placeholders
                + cur_num_spec_tokens
            ):
                # The request will generate a new token plus num_spec_tokens
                # in this scheduling step.
                request.num_output_placeholders += 1 + cur_num_spec_tokens
                # Add placeholders for the new tokens in spec_token_ids.
                # We will update the actual spec token ids in the worker process.
                request.spec_token_ids = [-1] * self.num_spec_tokens

        scheduler_output.pending_structured_output_tokens = (
            pending_structured_output_tokens
        )

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[int],
    ) -> tuple[list[int], bool]:
        if request.discard_latest_async_tokens:
            # If the request is force preempted in reset_prefix_cache, we
            # should discard the latest async token.
            request.discard_latest_async_tokens = False
            return [], False

        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )

        # Update the number of output placeholders.
        request.num_output_placeholders -= len(new_token_ids)
        assert request.num_output_placeholders >= 0

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped