# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_dummy_kv_cache,
    create_standard_kv_cache_spec,
    create_vllm_config,
)
from vllm.config import set_current_vllm_config
from vllm.v1.attention.backends.utils import PerLayerParameters
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE


class _FakeDycpGroup:

    def __init__(self, world_size: int = 2, rank_in_group: int = 0) -> None:
        self.world_size = world_size
        self.rank_in_group = rank_in_group


class _MockAttentionLayer:

    def __init__(self, device: torch.device):
        self._q_scale = torch.tensor(1.0, device=device)
        self._k_scale = torch.tensor(1.0, device=device)
        self._v_scale = torch.tensor(1.0, device=device)
        self._q_scale_float = 1.0
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0


def _make_gqa_test_config():
    return create_vllm_config(
        model_name="meta-llama/Meta-Llama-3-8B",
        max_model_len=64,
        block_size=16,
        num_gpu_blocks=256,
        max_num_seqs=8,
        max_num_batched_tokens=64,
    )


def _as_torch_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == "auto":
        return torch.float16
    return STR_DTYPE_TO_TORCH_DTYPE[dtype]


def _make_mixed_dycp_prefill_common_metadata(
    vllm_config,
    device: torch.device,
    num_dycp_reqs: int = 2,
):
    batch_spec = BatchSpec(
        seq_lens=[24, 20, 28],
        query_lens=[4, 3, 5],
    )
    common_attn_metadata = create_common_attn_metadata(
        batch_spec=batch_spec,
        block_size=vllm_config.cache_config.block_size,
        device=device,
    )
    common_attn_metadata.num_dycp_reqs = num_dycp_reqs
    return common_attn_metadata


def _make_qkv_and_cache(vllm_config, device: torch.device):
    num_heads = vllm_config.model_config.get_num_attention_heads(
        vllm_config.parallel_config
    )
    num_kv_heads = vllm_config.model_config.get_num_kv_heads(
        vllm_config.parallel_config
    )
    head_size = vllm_config.model_config.get_head_size()
    dtype = _as_torch_dtype(vllm_config.model_config.dtype)
    total_tokens = 12

    query = torch.arange(
        total_tokens * num_heads * head_size,
        dtype=dtype,
        device=device,
    ).reshape(total_tokens, num_heads, head_size)
    key = query[:, :num_kv_heads].clone()
    value = key.clone() + 3
    kv_cache = create_dummy_kv_cache(
        block_size=vllm_config.cache_config.block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
        device=device,
        num_blocks=8,
    )
    return query, key, value, kv_cache


def test_flash_attn_mixed_dycp_prefill_matches_split_sub_batches(monkeypatch):
    import vllm.distributed.parallel_state as parallel_state
    from vllm.v1.attention.backends import flash_attn as flash_attn_backend

    fake_group = _FakeDycpGroup()
    monkeypatch.setattr(parallel_state, "get_dycp_group", lambda: fake_group)
    monkeypatch.setattr(flash_attn_backend, "get_dycp_group", lambda: fake_group)

    def fake_flash_attn_varlen_func(q, *args, **kwargs):
        output = q + 1
        lse = torch.zeros(
            (q.shape[1], q.shape[0]),
            dtype=torch.float32,
            device=q.device,
        )
        return output, lse

    monkeypatch.setattr(
        flash_attn_backend,
        "flash_attn_varlen_func",
        fake_flash_attn_varlen_func,
        raising=False,
    )
    monkeypatch.setattr(
        flash_attn_backend,
        "cp_lse_ag_out_ar",
        lambda out, *args, **kwargs: out + 100,
    )

    vllm_config = _make_gqa_test_config()
    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)
    common_attn_metadata = _make_mixed_dycp_prefill_common_metadata(
        vllm_config,
        torch.device("cpu"),
    )

    builder = flash_attn_backend.FlashAttentionMetadataBuilder(
        kv_cache_spec,
        ["placeholder"],
        vllm_config,
        torch.device("cpu"),
    )
    builder.aot_schedule = False
    attn_metadata = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_attn_metadata,
    )
    assert attn_metadata._dycp_split is not None
    assert attn_metadata._dp_split is not None
    assert attn_metadata._dycp_split.num_actual_tokens == 7
    assert attn_metadata._dp_split.num_actual_tokens == 5

    query, key, value, kv_cache = _make_qkv_and_cache(
        vllm_config,
        torch.device("cpu"),
    )
    kv_cache = kv_cache.transpose(0, 1).contiguous()
    layer = _MockAttentionLayer(torch.device("cpu"))

    with set_current_vllm_config(vllm_config, check_compile=False):
        impl = flash_attn_backend.FlashAttentionImpl(
            num_heads=vllm_config.model_config.get_num_attention_heads(
                vllm_config.parallel_config
            ),
            head_size=vllm_config.model_config.get_head_size(),
            scale=1.0 / (vllm_config.model_config.get_head_size() ** 0.5),
            num_kv_heads=vllm_config.model_config.get_num_kv_heads(
                vllm_config.parallel_config
            ),
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            kv_sharing_target_layer_name="shared",
        )

    dycp_tokens = attn_metadata._dycp_split.num_actual_tokens
    expected_dycp = impl.forward(
        layer,
        query[:dycp_tokens],
        key[:dycp_tokens],
        value[:dycp_tokens],
        kv_cache,
        attn_metadata._dycp_split,
        output=torch.empty_like(query[:dycp_tokens]),
    )
    expected_dp = impl.forward(
        layer,
        query[dycp_tokens:],
        key[dycp_tokens:],
        value[dycp_tokens:],
        kv_cache,
        attn_metadata._dp_split,
        output=torch.empty_like(query[dycp_tokens:]),
    )
    expected = torch.cat([expected_dycp, expected_dp], dim=0)

    actual = impl.forward(
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=torch.empty_like(query),
    )
    torch.testing.assert_close(actual, expected)


def test_flashinfer_mixed_dycp_prefill_matches_split_sub_batches(monkeypatch):
    pytest.importorskip("flashinfer")

    import vllm.distributed.parallel_state as parallel_state
    from vllm.v1.attention.backends import flashinfer as flashinfer_backend

    fake_group = _FakeDycpGroup()
    monkeypatch.setattr(parallel_state, "get_dycp_group", lambda: fake_group)

    class FakePrefillWrapper:

        def __init__(self, bias: int, sm_scale: float):
            self.bias = bias
            self._window_left = -1
            self._logits_soft_cap = 0.0
            self._sm_scale = sm_scale
            self._causal = True

        def run(self, query, *args, **kwargs):
            kwargs["out"].copy_(query + self.bias)

    monkeypatch.setattr(
        flashinfer_backend,
        "BatchPrefillWithPagedKVCacheWrapper",
        FakePrefillWrapper,
    )

    def mock_get_per_layer_parameters(vllm_config, layer_names, impl_cls):
        head_size = vllm_config.model_config.get_head_size()
        return {
            layer_name: PerLayerParameters(
                window_left=-1,
                logits_soft_cap=0.0,
                sm_scale=1.0 / (head_size**0.5),
            )
            for layer_name in layer_names
        }

    monkeypatch.setattr(
        flashinfer_backend,
        "get_per_layer_parameters",
        mock_get_per_layer_parameters,
    )

    def fake_build_split_prefill_metadata(self, common_attn_metadata):
        bias = 100 if common_attn_metadata.num_dycp_reqs > 0 else 1
        q_dtype = _as_torch_dtype(self.model_config.dtype)
        return flashinfer_backend.FlashInferMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            q_data_type=q_dtype,
            slot_mapping=common_attn_metadata.slot_mapping,
            max_q_len=common_attn_metadata.max_query_len,
            max_q_len_prefill=common_attn_metadata.max_query_len,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table_tensor=common_attn_metadata.block_table_tensor,
            prefill_use_trtllm=False,
            decode_use_trtllm=False,
            num_decodes=0,
            num_decode_tokens=0,
            num_prefills=common_attn_metadata.num_reqs,
            num_prefill_tokens=common_attn_metadata.num_actual_tokens,
            use_cascade=False,
            prefill_wrapper=FakePrefillWrapper(bias=bias, sm_scale=self.sm_scale),
            num_dycp_reqs=common_attn_metadata.num_dycp_reqs,
        )

    monkeypatch.setattr(
        flashinfer_backend.FlashInferMetadataBuilder,
        "_build_split_prefill_metadata",
        fake_build_split_prefill_metadata,
    )

    vllm_config = _make_gqa_test_config()
    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)
    common_attn_metadata = _make_mixed_dycp_prefill_common_metadata(
        vllm_config,
        torch.device("cpu"),
    )

    with set_current_vllm_config(vllm_config, check_compile=False):
        builder = flashinfer_backend.FlashInferMetadataBuilder(
            kv_cache_spec,
            ["placeholder"],
            vllm_config,
            torch.device("cpu"),
        )
        attn_metadata = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )
        impl = flashinfer_backend.FlashInferImpl(
            num_heads=vllm_config.model_config.get_num_attention_heads(
                vllm_config.parallel_config
            ),
            head_size=vllm_config.model_config.get_head_size(),
            scale=1.0 / (vllm_config.model_config.get_head_size() ** 0.5),
            num_kv_heads=vllm_config.model_config.get_num_kv_heads(
                vllm_config.parallel_config
            ),
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            kv_sharing_target_layer_name=0,
        )

    assert attn_metadata._dycp_split is not None
    assert attn_metadata._dp_split is not None
    assert attn_metadata._dycp_split.num_actual_tokens == 7
    assert attn_metadata._dp_split.num_actual_tokens == 5

    query, key, value, kv_cache = _make_qkv_and_cache(
        vllm_config,
        torch.device("cpu"),
    )
    attn_metadata.q_data_type = query.dtype
    layer = _MockAttentionLayer(torch.device("cpu"))

    dycp_tokens = attn_metadata._dycp_split.num_actual_tokens
    expected_dycp = impl.forward(
        layer,
        query[:dycp_tokens],
        key[:dycp_tokens],
        value[:dycp_tokens],
        kv_cache,
        attn_metadata._dycp_split,
        output=torch.empty_like(query[:dycp_tokens]),
    )
    expected_dp = impl.forward(
        layer,
        query[dycp_tokens:],
        key[dycp_tokens:],
        value[dycp_tokens:],
        kv_cache,
        attn_metadata._dp_split,
        output=torch.empty_like(query[dycp_tokens:]),
    )
    expected = torch.cat([expected_dycp, expected_dp], dim=0)

    actual = impl.forward(
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=torch.empty_like(query),
    )
    torch.testing.assert_close(actual, expected)
