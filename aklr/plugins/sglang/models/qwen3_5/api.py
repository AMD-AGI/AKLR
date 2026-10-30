# Created by OpenAI
"""State-preserving integration at SGLang's complete GDN forward boundary.

One route serves the layer's EXTEND (prefill) and DECODE calls.
"""

from functools import cache
from types import SimpleNamespace

import torch

from aklr.core.contract.spec import PointSpec
from aklr.core.runtime.dispatch import DispatcherRegistry, FallbackReason, api
from aklr.plugins.sglang.hook import sglang_route

from .contract import (
    GDN_DECODE,
    GDN_DECODE_GROUP_FP8_QUANT,
    GDN_PREFILL_GROUP_FP8_QUANT,
)


@cache
def _dependencies():
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
    from sglang.srt.layers.quantization.fp8_utils import aiter_w8a8_block_fp8_linear
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    return SimpleNamespace(
        fp8_method=Fp8LinearMethod,
        consumer=aiter_w8a8_block_fp8_linear,
        extend=ForwardMode.EXTEND,
        decode=ForwardMode.DECODE,
    )


@cache
def _arch(device):
    return torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]


def _hardware_supported(hidden):
    return (
        isinstance(hidden, torch.Tensor)
        and hidden.device.type == "cuda"
        and torch.version.hip is not None
        and _arch(hidden.device) == "gfx950"
    )


def _tensor_matches(tensor, shape, dtype, device):
    return (
        isinstance(tensor, torch.Tensor)
        and tuple(tensor.shape) == tuple(shape)
        and tensor.dtype == dtype
        and tensor.device == device
        and tensor.is_contiguous()
    )


def _backend(batch):
    backend = getattr(batch, "attn_backend", None)
    if backend is None:
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend()
    return getattr(backend, "linear_attn_backend", None)


@cache
def _zero_bias(device):
    return torch.zeros(3072, dtype=torch.bfloat16, device=device)


@api(GDN_PREFILL_GROUP_FP8_QUANT)
def gdn_prefill_group_fp8_quant(
    projected_qkvz,
    projected_ba,
    conv_state,
    delta_state,
    cache_indices,
    cu_seqlens,
    has_initial_state,
    conv_weight,
    conv_bias,
    a_log,
    dt_bias,
    norm_weight,
    scale=128**-0.5,
    eps=1e-6,
    max_seq_len=2048,
    first_seq_len=0,
): ...


@api(GDN_DECODE_GROUP_FP8_QUANT)
def gdn_decode_group_fp8_quant(
    projected_qkvz,
    projected_ba,
    conv_state,
    delta_state,
    cache_indices,
    conv_weight,
    a_log,
    dt_bias,
    norm_weight,
    scale=128**-0.5,
    eps=1e-6,
): ...


@api(GDN_DECODE)
def gdn_decode(
    projected_qkvz,
    projected_ba,
    conv_state,
    delta_state,
    cache_indices,
    conv_weight,
    a_log,
    dt_bias,
    norm_weight,
    scale=128**-0.5,
    eps=1e-6,
): ...


def _partition(rows, lengths):
    """Scheduler metadata gives the contract its actual maximum, without GPU reads."""
    if not isinstance(lengths, (list, tuple)) or not lengths:
        return None
    if any(type(n) is not int or n <= 0 for n in lengths) or sum(lengths) != rows:
        return None
    maximum = max(lengths)
    if maximum not in GDN_PREFILL_GROUP_FP8_QUANT.attrs["max_seq_len"].values:
        return None
    return maximum, 10000 if tuple(lengths) == (10000, 6384) else 0


def _layer_supported(instance, deps):
    """The TP4 Qwen3.5 layer and block-FP8 output projection both ops serve."""
    layer = getattr(instance, "attn", None)
    norm = getattr(instance, "norm", None)
    out_proj = getattr(instance, "out_proj", None)
    method = getattr(out_proj, "quant_method", None)
    if layer is None or norm is None or type(method) is not deps.fp8_method:
        return False
    if (
        getattr(instance, "attn_tp_size", None) != 4
        or getattr(layer, "num_k_heads", None) != 4
        or getattr(layer, "num_v_heads", None) != 16
        or getattr(layer, "head_k_dim", None) != 128
        or getattr(layer, "head_v_dim", None) != 128
        or getattr(layer, "bias", None) is not None
        or getattr(layer, "activation", None) != "silu"
        or getattr(norm, "eps", None) != 1e-6
        or not getattr(norm, "norm_before_gate", False)
        or getattr(norm, "activation", None) != "swish"
        or getattr(norm, "group_size", None) is not None
        or not getattr(method, "block_quant", False)
        or getattr(method, "use_mxfp8", True)
        or getattr(method, "use_marlin", False)
        or getattr(method, "block_fp8_as_mxfp8", False)
        or hasattr(out_proj, "set_lora")
        or getattr(getattr(method, "quant_config", None), "weight_block_size", None) != [128, 128]
        or getattr(method, "w8a8_block_fp8_linear", None) is not deps.consumer
    ):
        return False
    qkv_weight = getattr(getattr(instance, "in_proj_qkvz", None), "weight", None)
    ba_weight = getattr(getattr(instance, "in_proj_ba", None), "weight", None)
    out_weight = getattr(out_proj, "weight", None)
    return (
        isinstance(qkv_weight, torch.Tensor)
        and isinstance(ba_weight, torch.Tensor)
        and tuple(qkv_weight.shape) == (5120, 4096)
        and tuple(ba_weight.shape) == (32, 4096)
        and isinstance(out_weight, torch.Tensor)
        and tuple(out_weight.shape) == (4096, 2048)
    )


def _prepare(instance, hidden, batch):
    """Check framework semantics before projections or state mutation."""
    if not _hardware_supported(hidden) or hidden.ndim != 2:
        return None
    deps = _dependencies()
    if getattr(batch, "forward_mode", None) != deps.extend:
        return None
    if torch.cuda.is_current_stream_capturing():
        return None
    rows = hidden.shape[0]
    lengths = getattr(batch, "extend_seq_lens_cpu", None)
    partition = _partition(rows, lengths)
    if partition is None or getattr(batch, "batch_size", None) != len(lengths):
        return None
    prefixes = getattr(batch, "extend_prefix_lens_cpu", None)
    if (
        not isinstance(prefixes, list)
        or len(prefixes) != len(lengths)
        or any(type(n) is not int or n < 0 for n in prefixes)
    ):
        return None
    if not _layer_supported(instance, deps):
        return None
    layer, norm = instance.attn, instance.norm
    backend = _backend(batch)
    metadata = getattr(backend, "forward_metadata", None)
    if (
        metadata is None
        or getattr(metadata, "has_mamba_track_mask", True)
        or getattr(metadata, "num_state_checkpoints", 0)
        or getattr(backend, "mis_metadata", None) is not None
    ):
        return None
    pool = getattr(backend, "req_to_token_pool", None)
    lookup = getattr(pool, "mamba2_layer_cache", None)
    if not callable(lookup):
        return None
    cache = lookup(layer.layer_id)
    convolution = getattr(cache, "conv", None)
    if not isinstance(convolution, (list, tuple)) or not convolution:
        return None
    conv_state, delta_state = convolution[0], getattr(cache, "temporal", None)
    if not isinstance(conv_state, torch.Tensor) or conv_state.ndim != 3:
        return None
    slots = conv_state.shape[0]
    if slots < len(lengths):
        return None
    device = hidden.device
    indices = getattr(metadata, "mamba_cache_indices", None)
    starts = getattr(metadata, "query_start_loc", None)
    prefix_tensor = getattr(batch, "extend_prefix_lens", None)
    tensors = (
        (hidden, (rows, 4096), torch.bfloat16),
        (indices, (len(lengths),), torch.int32),
        (starts, (len(lengths) + 1,), torch.int32),
        (prefix_tensor, (len(lengths),), torch.int32),
        (conv_state, (slots, 3072, 3), torch.bfloat16),
        (delta_state, (slots, 16, 128, 128), torch.float32),
        (getattr(layer, "conv_weights", None), (3072, 4), torch.bfloat16),
        (getattr(instance, "A_log", None), (16,), torch.float32),
        (getattr(instance, "dt_bias", None), (16,), torch.bfloat16),
        (getattr(norm, "weight", None), (128,), torch.bfloat16),
    )
    if not all(_tensor_matches(t, shape, dtype, device) for t, shape, dtype in tensors):
        return None
    # Buffer addressing in these specializations uses signed 32-bit byte offsets.
    if any(t.numel() * t.element_size() >= 2**31 for t in (conv_state, delta_state)):
        return None
    return partition, conv_state, delta_state, indices, starts, prefix_tensor


def _is_decode(hidden, batch):
    """Plain DECODE on supported hardware; AMD dependencies load only on gfx950."""
    return _hardware_supported(hidden) and getattr(batch, "forward_mode", None) == _dependencies().decode


def _prepare_decode(instance, hidden, batch):
    """Host metadata only: decode also runs at graph capture, so nothing is read
    from the device and nothing changes before an op serves."""
    deps = _dependencies()
    if hidden.ndim != 2 or not _layer_supported(instance, deps):
        return None
    layer, rows = instance.attn, hidden.shape[0]
    backend = _backend(batch)
    metadata = getattr(backend, "forward_metadata", None)
    pool = getattr(backend, "req_to_token_pool", None)
    lookup = getattr(pool, "mamba2_layer_cache", None)
    if (
        metadata is None
        or not callable(lookup)
        or not callable(getattr(backend, "_track_mamba_state_decode", None))
        # ReplaySSM keeps per-slot ring buffers that these ops do not update.
        or getattr(metadata, "replayssm_write_pos", None) is not None
        or getattr(metadata, "replayssm_force_flush", None) is not None
    ):
        return None
    cache = lookup(layer.layer_id)
    convolution = getattr(cache, "conv", None)
    if not isinstance(convolution, (list, tuple)) or len(convolution) != 1:
        return None
    if any(getattr(cache, n, None) is not None for n in ("replayssm_d", "replayssm_k", "replayssm_g")):
        return None
    conv_state, delta_state = convolution[0], getattr(cache, "temporal", None)
    if not isinstance(conv_state, torch.Tensor) or conv_state.ndim != 3:
        return None
    slots, device = conv_state.shape[0], hidden.device
    # Padded graph rows carry SGLang's -1 sentinel; the ops leave their state alone.
    indices = getattr(metadata, "mamba_cache_indices", None)
    tensors = (
        (hidden, (rows, 4096), torch.bfloat16),
        (indices, (rows,), torch.int32),
        (conv_state, (slots, 3072, 3), torch.bfloat16),
        (delta_state, (slots, 16, 128, 128), torch.float32),
        (getattr(layer, "conv_weights", None), (3072, 4), torch.bfloat16),
        (getattr(instance, "A_log", None), (16,), torch.float32),
        (getattr(instance, "dt_bias", None), (16,), torch.bfloat16),
        (getattr(instance.norm, "weight", None), (128,), torch.bfloat16),
    )
    if not all(_tensor_matches(t, shape, dtype, device) for t, shape, dtype in tensors):
        return None
    return backend, conv_state, delta_state, indices


def _decode_point(contract, rows, slots):
    return PointSpec(
        contract_id=contract.contract_id,
        axes={
            "n_tokens": rows,
            "n_slots": slots,
            "n_v_heads": 16,
            "key_dim": 128,
            "value_dim": 128,
            "conv_dim": 3072,
        },
        dtypes={n: s.dtype.any_of[0].value for n, s in contract.tensors().items()},
        attrs={},
        layout_variant_name="row_major",
    )


def _gdn_decode_forward(instance, hidden_states, hidden, batch):
    """The decode layer: SGLang's projections, then the FP8 op (its values go to
    out_proj pre-quantized) or else the BF16 op; SGLang's state tracking runs after
    either, as in its own decode."""
    prepared = _prepare_decode(instance, hidden, batch)
    if prepared is None:
        return FallbackReason.OUT_OF_CONTRACT
    backend, conv_state, delta_state, indices = prepared
    rows, slots = hidden.shape[0], conv_state.shape[0]
    registry = DispatcherRegistry.default()
    ops = [
        (contract, op)
        for contract, op in (
            (GDN_DECODE_GROUP_FP8_QUANT, gdn_decode_group_fp8_quant),
            (GDN_DECODE, gdn_decode),
        )
        if registry.recorder is not None
        or registry.dispatcher(contract).has_workload(_decode_point(contract, rows, slots))
    ]
    if not ops:
        return FallbackReason.NO_WORKLOAD
    qkvz, ba = instance._forward_input_proj(hidden_states)
    if not (
        _tensor_matches(qkvz, (rows, 5120), torch.bfloat16, hidden.device)
        and _tensor_matches(ba, (rows, 32), torch.bfloat16, hidden.device)
    ):
        # Projection is read-only; no live state has been changed.
        return FallbackReason.OUT_OF_CONTRACT
    layer = instance.attn
    for contract, op in ops:
        result = op(
            projected_qkvz=qkvz,
            projected_ba=ba,
            conv_state=conv_state,
            delta_state=delta_state,
            cache_indices=indices,
            conv_weight=layer.conv_weights,
            a_log=instance.A_log,
            dt_bias=instance.dt_bias,
            norm_weight=instance.norm.weight,
            scale=layer.head_k_dim**-0.5,
            eps=instance.norm.eps,
        )
        if not isinstance(result, FallbackReason):
            break
    else:
        return result
    backend._track_mamba_state_decode(batch, conv_state, delta_state, indices, layer.layer_id)
    if contract is GDN_DECODE_GROUP_FP8_QUANT:
        values, scales = result
        output, _ = instance.out_proj((values, scales))
    else:
        output, _ = instance.out_proj(result.reshape(rows, -1))
    return output


@sglang_route(
    "sglang.srt.models.qwen3_5.Qwen3_5GatedDeltaNet.forward",
    versions=">=0.5.22,<0.5.23",
)
def _gdn_forward(self, hidden_states, forward_batch):
    """Translate the framework state; dispatch owns implementation selection."""
    # AMD's fused AR+norm path carries a BF16 side-output and FP8 projections.
    # Keep the original tuple for SGLang's input-projection method.
    hidden = (
        hidden_states[0] if isinstance(hidden_states, tuple) and len(hidden_states) == 3 else hidden_states
    )
    if _is_decode(hidden, forward_batch):
        return _gdn_decode_forward(self, hidden_states, hidden, forward_batch)
    prepared = _prepare(self, hidden, forward_batch)
    if prepared is None:
        return FallbackReason.OUT_OF_CONTRACT
    partition, conv_state, delta_state, indices, starts, prefixes = prepared
    registry = DispatcherRegistry.default()
    dispatcher = registry.dispatcher(GDN_PREFILL_GROUP_FP8_QUANT)
    point = PointSpec(
        contract_id=GDN_PREFILL_GROUP_FP8_QUANT.contract_id,
        axes={
            "n_tokens": hidden.shape[0],
            "n_sequences": indices.numel(),
            "n_slots": conv_state.shape[0],
            "n_v_heads": 16,
            "key_dim": 128,
            "value_dim": 128,
            "conv_dim": 3072,
            "n_starts": starts.numel(),
        },
        dtypes={n: s.dtype.any_of[0].value for n, s in GDN_PREFILL_GROUP_FP8_QUANT.tensors().items()},
        attrs={"max_seq_len": partition[0], "first_seq_len": partition[1]},
        layout_variant_name="row_major",
    )
    if registry.recorder is None and not dispatcher.has_workload(point):
        return FallbackReason.NO_WORKLOAD
    qkvz, ba = self._forward_input_proj(hidden_states)
    rows, device = hidden.shape[0], hidden.device
    if not (
        _tensor_matches(qkvz, (rows, 5120), torch.bfloat16, device)
        and _tensor_matches(ba, (rows, 32), torch.bfloat16, device)
    ):
        # Projection is read-only; no live state has been changed.
        return FallbackReason.OUT_OF_CONTRACT
    result = gdn_prefill_group_fp8_quant(
        projected_qkvz=qkvz,
        projected_ba=ba,
        conv_state=conv_state,
        delta_state=delta_state,
        cache_indices=indices,
        cu_seqlens=starts,
        has_initial_state=prefixes > 0,
        conv_weight=self.attn.conv_weights,
        conv_bias=_zero_bias(device),
        a_log=self.A_log,
        dt_bias=self.dt_bias,
        norm_weight=self.norm.weight,
        scale=128**-0.5,
        eps=self.norm.eps,
        max_seq_len=partition[0],
        first_seq_len=partition[1],
    )
    if isinstance(result, FallbackReason):
        return result
    _, values, scales = result
    output, _ = self.out_proj((values, scales))
    return output
