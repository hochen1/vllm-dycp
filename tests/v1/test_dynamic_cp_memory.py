# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.worker.block_table as block_table_module
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    get_cp_kv_cache_model_len,
    get_cp_kv_cache_world_size,
)
from vllm.v1.worker.block_table import MultiGroupBlockTable

pytestmark = pytest.mark.cpu_test


def _make_config(max_model_len: int, dcp: int, pcp: int, dycp: int):
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=pcp,
            dp_per_domain=dycp,
        ),
    )


def _make_group(world_size: int, rank: int):
    return SimpleNamespace(world_size=world_size, rank_in_group=rank)


def test_full_attention_kv_budget_uses_all_cp_factors():
    config = _make_config(max_model_len=1025, dcp=2, pcp=3, dycp=4)
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float16,
    )

    assert get_cp_kv_cache_world_size(config) == 24
    assert get_cp_kv_cache_model_len(config) == cdiv(1025, 24)

    expected_pages = cdiv(cdiv(1025, 24), spec.block_size)
    assert spec.max_memory_usage_bytes(config) == (
        expected_pages * spec.page_size_bytes
    )


def test_multigroup_block_table_uses_dycp_for_request_width(monkeypatch):
    monkeypatch.setattr(
        block_table_module,
        "get_pcp_group",
        lambda: _make_group(world_size=2, rank=1),
    )
    monkeypatch.setattr(
        block_table_module,
        "get_dcp_group",
        lambda: _make_group(world_size=2, rank=0),
    )
    monkeypatch.setattr(
        block_table_module,
        "get_dycp_group",
        lambda: _make_group(world_size=3, rank=2),
    )

    table = MultiGroupBlockTable(
        max_num_reqs=2,
        max_model_len=1025,
        max_num_batched_tokens=128,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=[16],
        kernel_block_sizes=[16],
        cp_kv_cache_interleave_size=1,
    )[0]

    assert table.total_cp_world_size == 12
    assert table.total_cp_rank == 10
    assert table.max_num_blocks_per_req == cdiv(1025, 16 * 12)
