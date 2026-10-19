# Copyright 2026 FlagOS Contributors
# Author: liyan <yan.l@pku.edu.cn>
# SPDX-License-Identifier: Apache-2.0
"""Exercise production dispatch without allocating device tensors."""

import unittest
from unittest.mock import patch

from sglang_fl.dispatch.backends.vendor.ascend.patches import (
    minimax_m3_prefill_future as future,
)


class RecordingKernel:
    def __init__(self, name):
        self.name = name
        self.fn = self.signature

    @staticmethod
    def signature(payload, SCORE_TYPE, BLOCKS_PER_STEP):
        pass

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            return self.name, grid, args, kwargs

        return launch


class FutureDispatchTest(unittest.TestCase):
    def test_supported_and_fallback_calls_preserve_abi(self):
        original = RecordingKernel("causal")
        optimized = RecordingKernel("future")
        dispatch = future._FutureKernelDispatch(original)
        grid = object()
        payload = object()
        cases = [
            ((payload, "max", 1), {}, "future"),
            ((payload,), {"SCORE_TYPE": "max", "BLOCKS_PER_STEP": 1}, "future"),
            ((payload, "max"), {"BLOCKS_PER_STEP": 1}, "future"),
            ((payload, "max", 2), {}, "causal"),
            ((payload,), {"SCORE_TYPE": "max", "BLOCKS_PER_STEP": 4}, "causal"),
            ((payload, "lse", 1), {}, "causal"),
            ((payload,), {"SCORE_TYPE": "lse", "BLOCKS_PER_STEP": 1}, "causal"),
            ((payload,), {}, "causal"),
        ]
        with patch.object(future, "_fl_prefill_future_score_kernel", optimized):
            for args, kwargs, target in cases:
                with self.subTest(args=args, kwargs=kwargs):
                    self.assertEqual(
                        dispatch[grid](*args, **kwargs),
                        (target, grid, args, kwargs),
                    )

    def test_disabled_feature_leaves_kernel_unchanged(self):
        original = future.causal._fl_prefill_bnsd_score_kernel
        with patch.dict(
            "os.environ", {"SGLANG_FL_ASCEND_MINIMAX_PREFILL_FUTURE_SKIP": "0"}
        ):
            future.patch_ascend_minimax_prefill_future()
        self.assertIs(future.causal._fl_prefill_bnsd_score_kernel, original)


if __name__ == "__main__":
    unittest.main()
