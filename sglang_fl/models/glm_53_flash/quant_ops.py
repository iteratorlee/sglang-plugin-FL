"""Graph-safe FP32 dequantization for grouped INT8 expert GEMMs."""

import torch
import triton
import triton.language as tl


@triton.jit
def _dequantize_grouped_int32(
    accum,
    weight_scale,
    token_scale,
    counts,
    output,
    CHANNELS: tl.constexpr,
    EXPERTS: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
    BLOCK_EXPERTS: tl.constexpr,
):
    row = tl.program_id(0)
    expert_ids = tl.arange(0, BLOCK_EXPERTS)
    ends = tl.cumsum(tl.load(counts + expert_ids, expert_ids < EXPERTS, 0))
    expert = tl.sum(((row >= ends) & (expert_ids < EXPERTS)).to(tl.int32))
    channels = tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    valid = (channels < CHANNELS) & (expert < EXPERTS)
    value = tl.load(accum + row * CHANNELS + channels, valid, 0).to(tl.float32)
    channel_scale = tl.load(weight_scale + expert * CHANNELS + channels, valid, 0)
    activation_scale = tl.load(token_scale + row, expert < EXPERTS, 0)
    value = (value * activation_scale) * channel_scale
    tl.store(output + row * CHANNELS + channels, value, channels < CHANNELS)


def dequantize_grouped_int32(accum, weight_scale, token_scale, counts, output_dtype):
    """Use device-side expert counts, including empty experts and padded rows.

    CANN 8.5's fused GMM requires BF16 channel scales for BF16 output. Keep
    the checkpoint's FP32 scales until the final output conversion instead.
    Inputs are contiguous ND tensors; counts describe grouped token rows.
    """
    rows, channels = accum.shape
    output = torch.empty_like(accum, dtype=output_dtype)
    if rows:
        block_channels = min(triton.next_power_of_2(channels), 4096)
        _dequantize_grouped_int32[(rows, triton.cdiv(channels, block_channels))](
            accum,
            weight_scale,
            token_scale,
            counts,
            output,
            CHANNELS=channels,
            EXPERTS=weight_scale.shape[0],
            BLOCK_CHANNELS=block_channels,
            BLOCK_EXPERTS=triton.next_power_of_2(weight_scale.shape[0]),
            num_warps=1,
        )
    return output
