# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
from collections.abc import Callable
from functools import wraps

from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
    is_v1_kv_transfer_group,
)


def maybe_transfer_kv_layer(func: Callable) -> Callable:
    """Decorator that handles KV layer transfer prior and after execution of
    an attention layer, if enabled. Otherwise, the wrapper is a no-op.

    On entry: waits for the KV layer from the connector.
    On exit: saves the KV layer to the connector.
    """
    # Import at runtime to avoid circular dependency
    from vllm.attention.layer import get_attention_context

    # Inspect the signature ONCE when the decorator is applied.
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())

    # Find the index of 'layer_name' parameter.
    try:
        layer_name_index = param_names.index("layer_name")
    except ValueError as e:
        raise TypeError(
            f"Function {func.__name__} must have a 'layer_name' parameter"
        ) from e

    # For MLA attention: detect kv_c_normed and k_pe parameters.
    # These contain the pre-interleave-mask KV needed by PD connectors.
    _kv_c_normed_idx = (
        param_names.index("kv_c_normed") if "kv_c_normed" in param_names else -1
    )
    _k_pe_idx = param_names.index("k_pe") if "k_pe" in param_names else -1

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
            return func(*args, **kwargs)

        layer_name: str = args[layer_name_index]

        # Extract attention context (layer-specific metadata, layer, and kv_cache)
        attn_metadata, attn_layer, kv_cache = get_attention_context(layer_name)
        connector = get_kv_transfer_group()
        if attn_metadata is None or not connector.has_connector_metadata():
            return func(*args, **kwargs)

        # Wait for KV layer on entry
        connector.wait_for_layer_load(layer_name)

        # Execute the function
        result = func(*args, **kwargs)

        # Save KV cache layer on exit.
        # For MLA, pass raw pre-mask KV tensors so connectors can
        # capture complete KV before interleave mask discards data.
        extra_kw: dict = {}
        if _kv_c_normed_idx >= 0 and _kv_c_normed_idx < len(args):
            extra_kw["_raw_kv_c_normed"] = args[_kv_c_normed_idx]
        if _k_pe_idx >= 0 and _k_pe_idx < len(args):
            extra_kw["_raw_k_pe"] = args[_k_pe_idx]
        connector.save_kv_layer(layer_name, kv_cache, attn_metadata,
                                **extra_kw)

        return result

    return wrapper
