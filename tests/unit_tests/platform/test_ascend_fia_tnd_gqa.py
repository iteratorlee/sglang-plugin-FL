# Copyright 2026 FlagOS Contributors
"""Bounded FIA-TND contract tests for the final plugin worktree.

Run this with the final plugin worktree on ``PYTHONPATH``.  The import below
is intentionally the normal plugin import; no candidate path is embedded.
The NPU test mocks both KV-cache publication and FIA, so it never executes an
attention kernel.  It is skipped when an NPU is unavailable.
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    import torch
except ImportError:  # pragma: no cover - container test dependency
    torch = None

try:
    from sglang_fl.dispatch.backends.vendor.ascend.patches import fia_tnd_gqa
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
except ImportError:  # pragma: no cover - makes collection graceful off-container
    fia_tnd_gqa = None
    ForwardMode = None


T = 16384
Q_HEADS, KV_HEADS, D = 12, 2, 256
Q_WIDTH, V_WIDTH, PACKED_V_WIDTH = Q_HEADS * D, KV_HEADS * D, 7168


def _stock(
    self,
    q,
    k,
    v,
    layer,
    forward_batch,
    save_kv_cache=True,
    q_rope=None,
    k_rope=None,
    topk_indices=None,
    sinks=None,
    slopes=None,
):
    return "STOCK"


def _npu_available():
    if torch is None:
        return False
    try:
        import torch_npu  # noqa: F401

        return bool(torch.npu.is_available())
    except Exception:
        return False


def _layer(**overrides):
    values = dict(
        is_cross_attention=False,
        attn_type=SimpleNamespace(name="DECODER"),
        sliding_window_size=-1,
        logit_cap=0.0,
        tp_q_head_num=Q_HEADS,
        tp_k_head_num=KV_HEADS,
        tp_v_head_num=KV_HEADS,
        qk_head_dim=D,
        v_head_dim=D,
        scaling=0.0625,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _batch(device, *, seq=T, extend=T, metadata_device="cpu"):
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        seq_lens_cpu=torch.tensor([seq], dtype=torch.int32, device=metadata_device),
        extend_seq_lens_cpu=torch.tensor(
            [extend], dtype=torch.int32, device=metadata_device
        ),
        extend_prefix_lens_cpu=[0],
        encoder_lens=None,
        attn_cp_metadata=None,
        out_cache_loc=torch.tensor([17], dtype=torch.int64, device=device),
    )


@unittest.skipIf(fia_tnd_gqa is None, "SGLang plugin dependencies unavailable")
class TestAscendFiaTndGqa(unittest.TestCase):
    def test_default_off_and_cpu_ints(self):
        old = os.environ.pop("SGLANG_FL_FIA_TND_GQA", None)
        try:
            self.assertFalse(fia_tnd_gqa.patch_fia_tnd_gqa())
        finally:
            if old is not None:
                os.environ["SGLANG_FL_FIA_TND_GQA"] = old
        self.assertEqual(fia_tnd_gqa._cpu_ints(None), None)
        self.assertEqual(fia_tnd_gqa._cpu_ints([1, "2"]), [1, 2])
        self.assertEqual(
            fia_tnd_gqa._cpu_ints(torch.tensor([3, 4], dtype=torch.int32)), [3, 4]
        )
        self.assertIsNone(fia_tnd_gqa._cpu_ints(torch.tensor([5], device="meta")))
        self.assertIsNone(fia_tnd_gqa._cpu_ints(object()))

    def test_signature_guard_and_idempotence(self):
        from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
            AscendAttnBackend,
        )

        def incompatible(self, q):
            return "BAD"

        with mock.patch.object(AscendAttnBackend, "forward_extend", incompatible):
            with mock.patch.dict(os.environ, {"SGLANG_FL_FIA_TND_GQA": "1"}):
                self.assertFalse(fia_tnd_gqa.patch_fia_tnd_gqa())
                self.assertIs(AscendAttnBackend.forward_extend, incompatible)

        with mock.patch.object(AscendAttnBackend, "forward_extend", _stock):
            with mock.patch.dict(os.environ, {"SGLANG_FL_FIA_TND_GQA": "1"}):
                self.assertTrue(fia_tnd_gqa.patch_fia_tnd_gqa())
                wrapped = AscendAttnBackend.forward_extend
                self.assertTrue(getattr(wrapped, fia_tnd_gqa._MARKER, False))
                self.assertFalse(fia_tnd_gqa.patch_fia_tnd_gqa())
                self.assertIs(AscendAttnBackend.forward_extend, wrapped)

    @unittest.skipUnless(_npu_available(), "Ascend NPU unavailable")
    def test_packed_v_positive_and_eight_fallbacks(self):
        from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
            AscendAttnBackend,
        )

        device = torch.device("npu")
        q = torch.zeros((T, Q_WIDTH), dtype=torch.bfloat16, device=device)
        k = torch.zeros((T, KV_HEADS, D), dtype=torch.bfloat16, device=device)
        packed = torch.zeros((T, PACKED_V_WIDTH), dtype=torch.bfloat16, device=device)
        v = packed[:, -V_WIDTH:].view(T, KV_HEADS, D)
        self.assertEqual(tuple(v.stride()), (PACKED_V_WIDTH, D, 1))
        mask = torch.ones((2048, 2048), dtype=torch.bool, device=device).triu(1)
        mask = mask.contiguous()
        batch = _batch(device)
        kv_calls, fia_calls = [], []

        class KV:
            def set_kv_buffer(self, *args):
                kv_calls.append(args)

        backend = SimpleNamespace(
            use_fia=True,
            use_mla=False,
            use_alibi=False,
            attn_cp_size=1,
            is_dllm_model=False,
            fia_mask=mask,
        )
        batch.token_to_kv_pool = KV()
        layer = _layer()

        def fake_fia(q_, k_, v_, **kwargs):
            fia_calls.append((q_, k_, v_, kwargs))
            return (q_.new_zeros((q_.shape[0], q_.shape[1], v_.shape[-1])),)

        with mock.patch.object(AscendAttnBackend, "forward_extend", _stock):
            with mock.patch.dict(os.environ, {"SGLANG_FL_FIA_TND_GQA": "1"}):
                self.assertTrue(fia_tnd_gqa.patch_fia_tnd_gqa())
                wrapped = AscendAttnBackend.forward_extend
                with mock.patch.object(
                    torch.ops.npu,
                    "npu_fused_infer_attention_score",
                    fake_fia,
                    create=True,
                ):
                    result = wrapped(backend, q, k, v, layer, batch)
                    self.assertEqual(tuple(result.shape), (T, Q_WIDTH))
                    self.assertEqual(len(kv_calls), 1)
                    self.assertTrue(
                        all(
                            x is y
                            for x, y in zip(
                                kv_calls[0], (layer, batch.out_cache_loc, k, v)
                            )
                        )
                    )
                    self.assertEqual(len(fia_calls), 1)
                    call_q, call_k, call_v, kwargs = fia_calls[0]
                    self.assertEqual(call_q.data_ptr(), q.data_ptr())
                    self.assertEqual(call_k.data_ptr(), k.data_ptr())
                    self.assertEqual(call_v.data_ptr(), v.data_ptr())
                    self.assertEqual(tuple(call_v.stride()), (PACKED_V_WIDTH, D, 1))
                    self.assertIs(kwargs["atten_mask"], mask)
                    self.assertEqual(kwargs["input_layout"], "TND")
                    self.assertEqual(kwargs["actual_seq_lengths"], [T])
                    self.assertEqual(kwargs["actual_seq_lengths_kv"], [T])

                    bad_v = torch.zeros(
                        (T, 1024), dtype=torch.bfloat16, device=device
                    ).view(T, 4, D)[:, :KV_HEADS, :]
                    q_pad = torch.zeros(
                        (T, Q_WIDTH + 1), dtype=torch.bfloat16, device=device
                    )[:, :Q_WIDTH]
                    k_pad = torch.zeros(
                        (T, KV_HEADS, D + 1), dtype=torch.bfloat16, device=device
                    )[:, :, :D]
                    bad_mask_shape = SimpleNamespace(**vars(backend))
                    bad_mask_shape.fia_mask = torch.ones(
                        (1024, 1024), dtype=torch.bool, device=device
                    )
                    bad_mask_dtype = SimpleNamespace(**vars(backend))
                    bad_mask_dtype.fia_mask = torch.ones(
                        (2048, 2048), dtype=torch.float32, device=device
                    )
                    bad_heads = _layer(tp_k_head_num=4)
                    fallbacks = (
                        (backend, q, k, bad_v, layer, batch),
                        (backend, q_pad, k, v, layer, batch),
                        (backend, q, k_pad, v, layer, batch),
                        (backend, q, k, v, layer, _batch(device, seq=T + 1)),
                        (
                            backend,
                            q,
                            k,
                            v,
                            layer,
                            _batch(device, metadata_device=device),
                        ),
                        (bad_mask_shape, q, k, v, layer, batch),
                        (bad_mask_dtype, q, k, v, layer, batch),
                        (backend, q, k, v, bad_heads, batch),
                    )
                    for bs, qq, kk, vv, ll, bb in fallbacks:
                        self.assertEqual(wrapped(bs, qq, kk, vv, ll, bb), "STOCK")
                    self.assertEqual(len(kv_calls), 1)
                    self.assertEqual(len(fia_calls), 1)


if __name__ == "__main__":
    unittest.main()
