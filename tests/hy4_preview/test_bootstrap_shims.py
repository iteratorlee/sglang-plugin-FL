"""CPU-only regression tests for the shared, idempotent pool-init shim."""

import ast
import unittest
from pathlib import Path


SOURCE_DIR = Path(__file__).resolve().parents[2] / "sglang_fl/models/hy4_preview"


def load_pool_shim():
    # Execute the actual helper without importing Transformers or an NPU
    # runtime. Its only dependency is the pool class supplied by the caller.
    source = SOURCE_DIR / "bootstrap.py"
    tree = ast.parse(source.read_text())
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "patch_npu_mla_pool_init"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[function.name]


class PoolInitShimTest(unittest.TestCase):
    def setUp(self):
        self.patch_pool = load_pool_shim()
        self.calls = []
        calls = self.calls

        class Pool:
            def __init__(self, size, *, dtype):
                calls.append((self, size, dtype))

        self.pool_class = Pool

    def test_keyword_is_ignored_and_other_arguments_are_forwarded(self):
        self.patch_pool(self.pool_class)
        for extra in ({}, {"kv_cache_dim": None}, {"kv_cache_dim": 576}):
            pool = self.pool_class(128, dtype="bf16", **extra)
            self.assertEqual(self.calls[-1], (pool, 128, "bf16"))
        self.assertEqual(len(self.calls), 3)

    def test_repeated_bootstrap_worker_and_reload_calls_do_not_stack(self):
        self.patch_pool(self.pool_class)
        wrapped = self.pool_class.__init__
        for patch_pool in (self.patch_pool, load_pool_shim()):
            patch_pool(self.pool_class)
            self.assertIs(self.pool_class.__init__, wrapped)
        self.pool_class(128, dtype="bf16", kv_cache_dim=576)
        self.assertEqual(len(self.calls), 1)

    def test_original_errors_are_not_swallowed(self):
        self.patch_pool(self.pool_class)
        with self.assertRaises(TypeError):
            self.pool_class(128, dtype="bf16", unknown_keyword=True)
        self.assertEqual(self.calls, [])

    def test_both_entry_points_use_the_shared_helper(self):
        for name in ("bootstrap.py", "hy4.py"):
            tree = ast.parse((SOURCE_DIR / name).read_text())
            calls = [
                node for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "patch_npu_mla_pool_init"
            ]
            self.assertEqual(len(calls), 1, name)
            self.assertEqual(calls[0].args[0].id, "NPUMLATokenToKVPool")


if __name__ == "__main__":
    unittest.main(verbosity=2)
