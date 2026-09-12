"""GLM-scoped ModelSlim W8A8 support for SGLang 0.5.11 on Ascend.

Weights and per-token activations remain INT8 in both expert GEMMs. DeepEP
may communicate BF16; that is independent of the expert compute precision.
"""

from copy import deepcopy
import logging
import json
import os
from pathlib import Path

import torch

from sglang.srt.hardware_backend.npu.quantization.fused_moe_method_npu import (
    NPUW8A8Int8DynamicMoEMethod,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig
from sglang.srt.layers.quantization.modelslim.schemes.modelslim_w8a8_int8_moe import (
    ModelSlimW8A8Int8MoE,
)

from .checkpoint import canonical_weight_name
from .quant_ops import dequantize_grouped_int32


class GlmModelSlimConfig(ModelSlimConfig):
    """Copy the already-parsed config without repeating global RMSNorm patches."""

    def __init__(self, base: ModelSlimConfig):
        QuantizationConfig.__init__(self)
        self.quant_description = {
            canonical_weight_name(name): value
            for name, value in base.quant_description.items()
        }
        self.ignore = list(base.ignore)
        self.packed_modules_mapping = deepcopy(base.packed_modules_mapping)

    def get_linear_scheme(self, layer, prefix=None):
        scheme = super().get_linear_scheme(layer, prefix)
        if scheme is None:
            raise ValueError(f"Missing GLM ModelSlim linear scheme: {prefix}")
        return scheme

    def get_moe_scheme(self, layer, prefix):
        kind = self.quant_description.get(prefix + ".0.gate_proj.weight")
        if kind != "W8A8_DYNAMIC":
            raise ValueError(f"Unsupported GLM ModelSlim MoE scheme: {prefix}: {kind}")
        return _GlmW8A8MoEScheme(self)


class _GlmW8A8MoEScheme(ModelSlimW8A8Int8MoE):
    def __init__(self, quant_config):
        super().__init__(quant_config)
        self.kernel = GlmW8A8MoEMethod()


class GlmW8A8MoEMethod(NPUW8A8Int8DynamicMoEMethod):
    """DeepEP's grouped expert path with GLM's pre-SiLU clamp restored."""

    _audited_shapes = set()

    def apply(self, layer, dispatch_output):
        # The stock Standard-dispatch kernel lacks GLM's activation clamp.
        # This adaptation is intentionally scoped to the requested EP path.
        raise NotImplementedError(
            "GLM ModelSlim W8A8 requires --moe-a2a-backend deepep"
        )

    def process_weights_after_loading(self, layer):
        # ModelSlim's exported offsets are zero for symmetric W8A8. The
        # Ascend dynamic kernels do not consume asymmetric offsets.
        for name in ("w13_weight_offset", "w2_weight_offset"):
            if torch.count_nonzero(getattr(layer, name)).item():
                raise ValueError(f"GLM W8A8 requires symmetric weights: {name}")
        for name in ("w13_weight_scale", "w2_weight_scale"):
            scale = getattr(layer, name)
            if not (torch.isfinite(scale).all() & (scale > 0).all()).item():
                raise ValueError(f"Invalid GLM W8A8 scales: {name}")

        # CANN 8.5's clamped SwiGLU mode reads interleaved gate/up columns.
        # Repack once, before the stock transpose/NZ cast, including FP32
        # scales. This avoids a per-token layout copy and a lossy BF16 GMM1
        # intermediate before activation quantization.
        for name in ("w13_weight", "w13_weight_scale"):
            param = getattr(layer, name)
            width = param.shape[1]
            indices = (
                torch.arange(width, device="cpu")
                .reshape(2, width // 2)
                .T.flatten()
                .to(param.device)
            )
            # A 4D reshape/transpose leaves NCHW storage metadata on Ascend,
            # inflating the following NZ cast. Indexing keeps a 3D ND tensor.
            param.data = param.data.index_select(1, indices)
        super().process_weights_after_loading(layer)

    def apply_without_routing_weights(
        self,
        layer,
        hidden_states,
        hidden_states_scale,
        group_list_type,
        group_list,
        output_dtype,
    ):
        if group_list_type != 1:
            raise ValueError("GLM DeepEP W8A8 expects per-expert token counts")
        dispatch_dtype = str(hidden_states.dtype)
        if hidden_states.dtype != torch.int8:
            hidden_states, hidden_states_scale = torch.ops.npu.npu_dynamic_quant(
                hidden_states
            )
        if hidden_states_scale is None:
            raise ValueError("INT8 expert activations require per-token scales")

        if any(
            t.dtype != torch.int8
            for t in (hidden_states, layer.w13_weight, layer.w2_weight)
        ):
            raise TypeError(
                "GLM W8A8 expert GEMMs require INT8 activations and weights"
            )
        shape = tuple(hidden_states.shape)
        if shape not in self._audited_shapes and len(self._audited_shapes) < 4:
            self._audited_shapes.add(shape)
            logging.getLogger(__name__).info(
                "GLM W8A8 compute audit: x=%s %s, w13=%s %s, w2=%s %s, "
                "GEMM1/GEMM2=int8*int8->int32, output_scale=%s",
                shape,
                hidden_states.dtype,
                tuple(layer.w13_weight.shape),
                layer.w13_weight.dtype,
                tuple(layer.w2_weight.shape),
                layer.w2_weight.dtype,
                layer.w2_weight_scale.dtype,
            )

            audit_dir = os.getenv("SGLANG_FL_GLM53_AUDIT_DIR")
            if audit_dir:
                directory = Path(audit_dir)
                directory.mkdir(parents=True, exist_ok=True)
                with (directory / f"experts-{os.getpid()}.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "source": __file__,
                                "input_shape": shape,
                                "input_dtype": str(hidden_states.dtype),
                                "dispatch_dtype": dispatch_dtype,
                                "w13_shape": list(layer.w13_weight.shape),
                                "w13_dtype": str(layer.w13_weight.dtype),
                                "w2_shape": list(layer.w2_weight.shape),
                                "w2_dtype": str(layer.w2_weight.dtype),
                                "scale_dtype": str(layer.w2_weight_scale.dtype),
                                "gemm1": "int8*int8->int32",
                                "gemm2": "int8*int8->int32",
                                "triton_parallel_mode": os.getenv(
                                    "TRITON_ALL_BLOCKS_PARALLEL"
                                ),
                            }
                        )
                        + "\n"
                    )

        gate_up_int32 = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[layer.w13_weight],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=torch.int32,
        )[0]
        quantized, scale = torch.ops.npu.npu_dequant_swiglu_quant(
            gate_up_int32,
            weight_scale=layer.w13_weight_scale,
            activation_scale=hidden_states_scale.reshape(-1),
            group_index=group_list,
            activate_left=True,
            quant_mode=1,
            swiglu_mode=1,
            clamp_limit=layer._sglang_fl_swiglu_limit,
            glu_alpha=1.0,
            glu_bias=0.0,
        )
        down_int32 = torch.ops.npu.npu_grouped_matmul(
            x=[quantized],
            weight=[layer.w2_weight],
            split_item=2,
            group_list_type=group_list_type,
            group_type=0,
            group_list=group_list,
            output_dtype=torch.int32,
        )[0]
        return dequantize_grouped_int32(
            down_int32, layer.w2_weight_scale, scale, group_list, output_dtype
        )
