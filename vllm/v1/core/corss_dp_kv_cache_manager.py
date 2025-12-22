import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, overload, NewType

from vllm.distributed.kv_events import KVCacheEvent
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import (
    CrossAttentionManager,
    FullAttentionManager,
    get_manager_for_kv_cache_spec,
)

logger = init_logger(__name__)

BlockHash = NewType("BlockHash", bytes)

class DPBlockPool(BlockPool):
    def __init__(
        self,
        dcp_rank: int,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        self.dcp_rank = dcp_rank
        super.__init__(
            num_gpu_blocks,
            enable_caching,
            hash_block_size,
            enable_kv_cache_events,
            metrics_collector,
        )
    
    def get_dcp_rank_id(self):
        return self.dcp_rank

class CorssDPKVCacheCoordinatorNoPrefixCache:
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int,
        pcp_world_size: int,
        hash_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching

        self.block_pools = [
            DPBlockPool(
            rank,
            kv_cache_config.num_blocks,
            enable_caching,
            hash_block_size,
            enable_kv_cache_events,
            metrics_collector,
        ) for rank in range(dcp_world_size)]

        self.use_eagle = use_eagle

        self.corss_dp_single_type_managers = [
            tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                block_pool=self.block_pools[rank],
                kv_cache_group_id=i,
                dcp_world_size=1,
                pcp_world_size=1,
            )
            for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups)
        ) for rank in range(dcp_world_size)
        ]

    def get_num_blocks_to_allocate(self) -> int:
        pass

    def save_new_computed_blocks(self) -> None:
        pass

    def allocate_new_blocks(self) -> None:
        pass

    def cache_blocks(self) -> None:
        pass

    def free(self, request_id: str) -> None:
        pass

    def remove_skipped_blocks(self, request_id: str, num_compted_tokens: int) -> None:
        pass

    def get_blocks(self, request_id: str) -> tuple[list[KVCacheBlock], ...]:
        pass

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        return [0] * self.num_single_type_manager

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(self.num_single_type_manager)
        )
        return blocks, 0
    

class CorssDPKVCacheManager:
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        hash_block_size: int,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ) -> None:
        self.max_model_len = max_model_len

        self.enable_caching = enable_caching
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        self.metrics_collector = metrics_collector
        self.dcp_size = dcp_world_size
        self.pcp_size = pcp_world_size

        # FIXME: make prefix cache stats conditional on log_stats. We still need
        # this comment because when the log stats is enabled there are still
        # potential configs we could expose in the future.
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        self.coordinator = CorssDPKVCacheCoordinatorNoPrefixCache(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            use_eagle=self.use_eagle,
            enable_caching=self.enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=self.metrics_collector,
        )

        # A corss dp manager manages the dp number block pool
        self.block_pools = self.coordinator.block_pools
        assert len(self.coordinator.block_pools) == dcp_world_size

        self.kv_cache_config = kv_cache_config

        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )
    
    @property
    def usage(self, dcp_rank) -> float:
        """Get the KV cache usage of specfic DP ranks

        Returns:
            The KV cache usage of all DP ranks
        """
        return self.block_pools[dcp_rank].get_usage()


    def make_prefix_cache_stats(self) -> PrefixCacheStats | None:
        raise NotImplementedError

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """ 
        The decode instance can ignore this?

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
        """

        if not self.enable_caching or request.skip_reading_prefix_cache:
            return self.empty_kv_cache_blocks, 0
        
    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
    ) -> KVCacheBlocks | None:
        
        if num_new_tokens == 0:
            raise ValueError("num_new_tokens must be greater than 0")

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = self.empty_kv_cache_blocks.blocks
        
        self.coordinator.remove_skipped_blocks(
            request.request_id, request.num_computed_tokens
        )

        num_computed_tokens = request.num_computed_tokens + num_new_computed_tokens
        num_tokens_need_slot = min(
            num_computed_tokens + num_new_tokens + num_lookahead_tokens,
            self.max_model_len,
        )

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
        )

        raise NotImplementedError

    def free(self, request: Request) -> None:
        raise NotImplementedError
    
    def get_blocks(self, request_id: str) -> list[KVCacheBlocks]:
        """Get the blocks of a request."""
        raise NotImplementedError
        return self.create_kv_cache_blocks(self.coordinator.get_blocks(request_id))
    
    
    def get_block_ids(self, request_id: str) -> list[tuple[list[int], ...]]:
        """Get the block ids of a request."""
        raise NotImplementedError
        return self.get_blocks(request_id).get_block_ids()
    
    def create_kv_cache_blocks(
        self, blocks: tuple[list[KVCacheBlock], ...]
    ) -> KVCacheBlocks:
        raise NotImplementedError
        # Only create new KVCacheBlocks for non-empty blocks
        return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks
    
    


    


