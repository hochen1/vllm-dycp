from functools import total_ordering
import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from threading import local
from typing import Literal, overload, NewType

from vllm.distributed.kv_events import KVCacheEvent
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.block_pool import BlockPool
from vllm.utils.math_utils import cdiv
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
        super().__init__(
            num_gpu_blocks,
            enable_caching,
            hash_block_size,
            enable_kv_cache_events,
            metrics_collector,
        )
    
    def get_dcp_rank_id(self):
        return self.dcp_rank

class CrossDPKVCacheCoordinatorNoPrefixCache:
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
        self.block_size = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size

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

    def _distribute_tokens_to_dcp_ranks(
        self,
        world_size: int,
        num_tokens: int) -> list[int]:

        total_blocks_required = cdiv(num_tokens, self.block_size)
        remain_tokens = num_tokens % self.block_size

        quota = total_blocks_required // world_size
        remainder = total_blocks_required % world_size

        results = []
        for idx in range(world_size):
            local_blocks = quota + (1 if idx < remainder else 0)
            if remainder - 1 == idx:
                local_tokens = (local_blocks - 1) * self.block_size + remain_tokens
            else:
                local_tokens = local_blocks * self.block_size
            results.append(local_tokens)
        
        return results

    def _allocate_blocks_to_dcp_ranks(
        self,
        world_size: int,
        num_tokens: int,
    ) -> list[list[int]]:
        """
        Map hash block IDs to DCP ranks in a round-robin fashion.
        
        Example with 10 blocks, 3 ranks:
        - rank 0: blocks [0, 3, 6, 9]
        - rank 1: blocks [1, 4, 7]
        - rank 2: blocks [2, 5, 8]
        
        Args:
            world_size: Number of DCP ranks.
            num_tokens: Total number of tokens.
        
        Returns:
            A list of lists, where each inner list contains the block IDs
            assigned to that rank. Format: [[rank0_blocks], [rank1_blocks], ...]
        """
        total_blocks = num_tokens // self.block_size

        # Initialize list of lists for each rank
        blocks_per_rank: list[list[int]] = [[] for _ in range(world_size)]
        
        # Distribute blocks in round-robin fashion
        # Block i goes to rank (i % world_size)
        for block_id in range(total_blocks):
            rank = block_id % world_size
            blocks_per_rank[rank].append(block_id)
        
        return blocks_per_rank

            
    def get_num_blocks_to_allocate(
        self,
        dcp_ranks: list[int],
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
    ) -> int:
        total_blocks = 0
        num_kv_cache_groups = len(self.kv_cache_config.kv_cache_groups)
        
        results = self._distribute_tokens_to_dcp_ranks(len(dcp_ranks), num_tokens)

        for id, rank in enumerate(dcp_ranks):
            if rank >= len(self.corss_dp_single_type_managers):
                raise ValueError(f"rank {rank} is out of range")

            rank_manager = self.corss_dp_single_type_managers[rank]

            for i in range(num_kv_cache_groups):

                total_blocks += rank_manager[i].get_num_blocks_to_allocate(
                    request_id, results[id], new_computed_blocks[i]
                )

        return total_blocks

    def save_new_computed_blocks(
        self, request_id: str, new_computed_blocks: tuple[Sequence[KVCacheBlock], ...]
    ) -> None:
        for rank_managers in self.corss_dp_single_type_managers:
            for i, manager in enumerate(rank_managers):
                assert len(new_computed_blocks[i]) == 0

    def allocate_new_blocks(
        self, dcp_ranks: list[int], request_id: str, num_tokens: int, num_encoder_tokens: int = 0
    ) -> list[tuple[list[KVCacheBlock], ...]]:
        
        blocks = []
        results = self._distribute_tokens_to_dcp_ranks(len(dcp_ranks), num_tokens)
        print(f"results={results}", flush=True)
        for idx, rank in enumerate(dcp_ranks):
            blocks.append(
                tuple(
                    manager.allocate_new_blocks(
                        request_id,
                        results[idx]
                    ) for manager in self.corss_dp_single_type_managers[rank]
                )
            )
        return blocks

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        raise NotImplementedError

    def free(self, request_id: str) -> None:
        """
        Free the blocks for the request across all DCP ranks.

        Args:
            request_id: The request ID.
        """
        # Free blocks on all DCP ranks (the request might have blocks on any rank)
        for rank in range(len(self.corss_dp_single_type_managers)):
            rank_managers = self.corss_dp_single_type_managers[rank]
            
            for manager in rank_managers:
                manager.free(request_id)

    def remove_skipped_blocks(self, dcp_ranks: list[int], request_id: str, num_compted_tokens: int) -> None:
        for rank in dcp_ranks:
            if rank >= len(self.corss_dp_single_type_managers):
                raise ValueError(f"rank {rank} is out of range")
            
            rank_managers = self.corss_dp_single_type_managers[rank]
            
            for manager in rank_managers:
                manager.remove_skipped_blocks(request_id, num_compted_tokens)

    def get_blocks(self, request_id: str) -> list[tuple[list[KVCacheBlock], ...]]:
        """
        Get the blocks for the request from all DCP ranks.

        Args:
            request_id: The request ID.

        Returns:
            A list of tuples, where each tuple contains the blocks for each
            KV cache group on that DCP rank.
            Format: [rank_0_blocks, rank_1_blocks, ...]
            where each rank_blocks is a tuple of lists, one per KV cache group.
        """
        blocks_by_rank = []
        
        for rank_managers in self.corss_dp_single_type_managers:
            rank_blocks = []
            
            for manager in rank_managers:
                # Get blocks for this manager (group)
                manager_blocks = manager.req_to_blocks.get(request_id, [])
                rank_blocks.append(manager_blocks)
            
            if any(rank_blocks) is not None:
                blocks_by_rank.append(tuple(rank_blocks))
        
        return blocks_by_rank

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

class CrossDPKVCacheManager:
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

        self.coordinator = CrossDPKVCacheCoordinatorNoPrefixCache(
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
        
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        
        # A corss dp manager manages the dp number block pool
        self.block_pools = self.coordinator.block_pools
        assert len(self.coordinator.block_pools) == dcp_world_size

        self.kv_cache_config = kv_cache_config

        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )
    
    @property
    def usage(self) -> float:
        """Get the KV cache usage of specfic DP ranks

        Returns:
            The KV cache usage of all DP ranks
        """
        return self.block_pools[0].get_usage()


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
        dcp_ranks: list[int],
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
    ) -> list[KVCacheBlocks] | None:
        """
            return:
                len(list[KVCacheBlocks]) = len(dcp_ranks)
        """
        if len(dcp_ranks) > self.dcp_size:
            raise ValueError("dcp_ranks can not greater than dcp_size")
        
        if len(dcp_ranks) != 1 and len(dcp_ranks) != self.dcp_size:
            raise NotImplementedError

        if num_new_tokens == 0:
            raise ValueError("num_new_tokens must be greater than 0")

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = self.empty_kv_cache_blocks.blocks
        
        self.coordinator.remove_skipped_blocks(
            dcp_ranks, request.request_id, request.num_computed_tokens
        )

        num_computed_tokens = request.num_computed_tokens + num_new_computed_tokens
        num_tokens_need_slot = min(
            num_computed_tokens + num_new_tokens + num_lookahead_tokens,
            self.max_model_len,
        )

        # Now the get_num_blocks_to_allocate return a int, it seems wrong, 
        # it should return a list[int], len(list[int]) = len(dcp_rank)
        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            dcp_ranks=dcp_ranks,
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
        )

        # This condition should be rewrite later for fine grained corss dp block allocation
        if all([num_blocks_to_allocate > block_pool.get_num_free_blocks() for block_pool in self.block_pools]):
            # Cannot allocate new blocks
            return None

    
        assert not any(new_computed_block_list), (
            "Computed blocks should be empty when prefix caching is disabled"
        )

        if new_computed_block_list is not self.empty_kv_cache_blocks.blocks:
            # Append the new computed blocks to the request blocks until now to
            # avoid the case where the new blocks cannot be allocated.

            # This should not be callled
            self.coordinator.save_new_computed_blocks(
                request.request_id, new_computed_block_list
            )

        new_blocks = self.coordinator.allocate_new_blocks(
            dcp_ranks=dcp_ranks,
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            num_encoder_tokens=num_encoder_tokens,
        )
        
        assert len(new_blocks) == len(dcp_ranks), "the size of new_blocks should be equal to size of dcp_ranks"

        if not self.enable_caching or delay_cache_blocks:
            # This condition is always true in this version
            return self.create_kv_cache_blocks(new_blocks)


    def free(self, request: Request) -> None:
        self.coordinator.free(request.request_id)

    def get_blocks(self, request: Request) -> list[KVCacheBlocks]:
        """Get the blocks of a request."""
        blocks_all = self.coordinator.get_blocks(request.request_id)   # List[Any]
        wanted = [blocks_all[i] for i in request.dcp_ranks if 0 <= i < len(blocks_all)]
        blocks = self.create_kv_cache_blocks(wanted)
        return blocks
    
    
    def get_block_ids(self, request: Request) -> list[tuple[list[int], ...]]:
        """Get the block ids of a request."""
        blocks = self.get_blocks(request)
        block_ids = [block.get_block_ids() for block in blocks]
        return block_ids

    def create_kv_cache_blocks(
        self, corss_blocks: list[tuple[list[KVCacheBlock], ...]]
    ) -> list[KVCacheBlocks]:
        # Only create new KVCacheBlocks for non-empty blocks
        kv_cache_blocks = [KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks for blocks in corss_blocks]
        # the len(kv_cache_blocks) == len(dcp_ranks) of the request
        return kv_cache_blocks
    
    def take_events(self):
        """
         Temparily ignore this
        """
        return []
    
    def make_prefix_cache_stats(self) -> PrefixCacheStats | None:
        """Get (and reset) the prefix cache stats.

        Returns:
            The current prefix caching stats, or None if logging is disabled.
        """
        if not self.log_stats:
            return None
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats