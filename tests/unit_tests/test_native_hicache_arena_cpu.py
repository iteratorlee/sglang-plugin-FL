"""Small real-CPU-storage tests, using AST-extracted audited SG startup code.

Run with conda torch and TORCH_DEVICE_BACKEND_AUTOLOAD=0, no pytest/GPU.
SGLANG_FL_ARENA_TEST_SOURCE points to the audited SG repository (test only).
Physical NPU pinning is deliberately NOT simulated as evidence of pinning.
"""
import ast
import gc
import importlib.util
import logging
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock
import weakref

import numpy as np
import torch

from sglang_fl import native_hicache_arena as policy
from sglang_fl import native_allocator_warmup

ROOT = Path(os.environ['SGLANG_FL_ARENA_TEST_SOURCE']).resolve()
sys.path.insert(0, str(ROOT / 'test/registered/unit/mem_cache'))
from test_glm53_early_host_arena_cpu import TestGLM53EarlyHostArenaCPU as SGFixture


class Namespace:
    def __init__(self, namespace):
        object.__setattr__(self, 'namespace', namespace)

    def __getattr__(self, key):
        try:
            return self.namespace[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key, value):
        self.namespace[key] = value


def mamba_module(base):
    path = ROOT / 'python/sglang/srt/mem_cache/pool_host/mamba.py'
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'MambaPoolHost')
    ns = dict(torch=torch, np=np, threading=threading, HostKVCache=base,
              logger=logging.getLogger(__name__), _is_npu=True,
              synchronized=lambda f: f, get_allocator_from_storage=lambda _: None,
              host_memory_budget_bytes=lambda: 1 << 30,
              sync_fixed_hicache_size=lambda size, _: size,
              ALLOC_MEMORY_FUNCS={'npu': mock.Mock(side_effect=AssertionError('late allocation'))})
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), ns)
    return Namespace(ns)


class ArenaTests(unittest.TestCase):
    def test_real_views_bounds_alignment_zero_and_lifetime(self):
        arena = torch.zeros(1024, dtype=torch.uint8)
        token = arena.untyped_storage()._weak_ref()
        self.addCleanup(torch.UntypedStorage._free_weak_ref, token)
        ref = lambda: None if torch.UntypedStorage._expired(token) else True
        specs = [((3, 7), torch.float32), ((0, 9), torch.float32),
                 ((2, 3), torch.bfloat16), ((1, 5), torch.bfloat16)]
        views, end = policy._typed_views(arena, 129, specs, pinned=False)
        self.assertLessEqual(end, arena.numel())
        for v in views:
            self.assertTrue(v.is_contiguous())
            self.assertEqual(v.untyped_storage()._cdata, arena.untyped_storage()._cdata)
            self.assertEqual(v.storage_offset() * v.element_size() % 64, 0)
        views[0].fill_(3)
        self.assertTrue(torch.all(arena[192:276].view(torch.float32) == 3))
        self.assertTrue(torch.all(arena[:129] == 0))
        del arena
        gc.collect()
        self.assertIsNotNone(ref())
        del views, v
        gc.collect()
        self.assertIsNone(ref())
        with self.assertRaisesRegex(RuntimeError, 'not pinned'):
            policy._typed_views(torch.zeros(16, dtype=torch.uint8), 0, [], pinned=True)
        # A narrow tensor has physically larger storage: bounds MUST use numel.
        narrow = torch.zeros(1024, dtype=torch.uint8)[:64]
        with self.assertRaisesRegex(RuntimeError, 'exhausted'):
            policy._typed_views(narrow, 64, [((1,), torch.float32)], pinned=False)

    def test_exact_production_bytes(self):
        self.assertEqual(policy._next_power2(64_000_000_000), 68_719_476_736)
        self.assertEqual(policy._next_power2(1 << 36), 1 << 36)
        end = 56_817_745_920 + 6_642_728_960 + 544_911_360
        self.assertEqual(end, 64_005_386_240)
        self.assertEqual((1 << 36) - end, 4_714_090_496)

    def test_source_guard_disabled_and_deferred_idempotence(self):
        root = ROOT / 'python/sglang'
        policy._validate_source(root)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            for relative in policy.SOURCE_HASHES:
                dst = target / relative
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes((root / relative).read_bytes())
            (target / next(iter(policy.SOURCE_HASHES))).write_text('# changed')
            with self.assertRaisesRegex(RuntimeError, 'Unaudited'):
                policy._validate_source(target)
        before = list(sys.meta_path)
        with mock.patch.dict(os.environ, {policy.FLAG: '0'}), mock.patch.object(
            native_allocator_warmup, '_source_root', side_effect=AssertionError('must not inspect SG')
        ):
            self.assertEqual(policy.configure()['status'], 'disabled')
        self.assertEqual(sys.meta_path, before)
        with mock.patch.dict(os.environ, {policy.FLAG: '1'}), mock.patch.object(
            native_allocator_warmup, '_source_root', return_value=root
        ), mock.patch.object(policy, '_finder', None):
            try:
                self.assertEqual(policy.configure()['status'], 'deferred')
                finder = policy._finder
                self.assertEqual(policy.configure()['status'], 'deferred')
                self.assertEqual(sys.meta_path.count(finder), 1)
                self.assertIsNone(finder.find_spec('anything_else'))
                for name in finder.targets:
                    self.assertNotIn(name, sys.modules)
                calls = []
                # Exercise real importlib deferred loading on a tiny module;
                # actual guarded SG code is exercised by LifecycleTests below.
                with tempfile.TemporaryDirectory() as directory:
                    p = Path(directory) / 'glm53.py'
                    p.write_text('answer = 42\n')
                    spec = importlib.util.spec_from_file_location(policy.PREFIX + 'glm53', p)
                    finder.targets[spec.name] = lambda module, state: calls.append(module.answer)
                    spec.loader = policy._Loader(spec.loader, finder, spec.name)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self.assertEqual(calls, [42])
            finally:
                sys.meta_path[:] = before


class LifecycleTests(unittest.TestCase):
    setUp = SGFixture.setUp
    make_host = SGFixture.make_host

    def prepare(self):
        module = Namespace(self.core)
        state = policy._State()
        # Real CPU tensors; explicitly bypass ONLY pin verification in tests.
        original_check, original_views = policy._check_arena, policy._typed_views
        for patch in (
            mock.patch.object(policy, '_check_arena', lambda a, **kw: original_check(a, pinned=False)),
            mock.patch.object(policy, '_typed_views', lambda a, o, s: original_views(a, o, s, pinned=False)),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        policy._install_glm(module, state)
        saved = module._reserve_early_host_arena
        policy._install_glm(module, state)
        self.assertIs(module._reserve_early_host_arena, saved)
        mamba = mamba_module(self.core['HostKVCache'])
        policy._install_mamba(mamba, state)
        with self.scope(self.runner):
            record = self.core['_early_host_arena']
            token = record.arena.untyped_storage()._weak_ref()
            self.addCleanup(torch.UntypedStorage._free_weak_ref, token)
            ref = lambda: None if torch.UntypedStorage._expired(token) else True
        host = self.make_host()
        self.assertIsNone(record.arena)  # Actual SG commit/discard ran.
        self.assertIsNone(self.core['_early_host_arena'])
        self.assertIsNotNone(ref())
        self.assertIs(state.pool(), host)
        self.assertEqual(len(self.allocator.calls), 1)
        self.assertEqual(self.allocator.calls[0][0], (1 << 25,))
        return module, state, mamba, host, ref

    def device_pool(self, temporal_shape=(3, 4), index=3):
        def tensor(shape, dtype):
            return SimpleNamespace(shape=shape, dtype=dtype,
                                   device=SimpleNamespace(type='npu', index=index))
        return SimpleNamespace(device='npu', size=3, num_mamba_layers=2,
            mamba_cache=SimpleNamespace(temporal=tensor((2, 3) + temporal_shape, torch.float32),
                conv=[tensor((2, 3, 2, 3), torch.bfloat16), tensor((2, 3, 5), torch.bfloat16)]))

    def test_actual_sg_commit_mamba_multiple_conv_destroy(self):
        module, state, mamba, host, ref = self.prepare()
        pool = mamba.MambaPoolHost(self.device_pool(), 2, 0, layout='page_first_direct')
        self.assertIsNone(host._fl_hybrid_owner)
        self.assertIsNone(state.pool())
        self.assertEqual(len(self.allocator.calls), 1)
        self.assertEqual(pool.temporal_buffer.shape, (7, 2, 1, 3, 4))
        self.assertEqual(len(pool.conv_buffer), 2)
        ptr = host.kv_buffer.untyped_storage()._cdata
        for v in pool.kv_buffer:
            self.assertEqual(v.untyped_storage()._cdata, ptr)
            self.assertTrue(v.is_contiguous())
            self.assertGreaterEqual(v.storage_offset() * v.element_size(), host._packed_host_arena.numel())
        del v
        host.destroy()
        gc.collect()
        self.assertIsNotNone(ref())  # Mamba alone still owns storage.
        pool.destroy()
        pool.destroy()
        gc.collect()
        self.assertIsNone(ref())

    def test_zero_temporal_and_reverse_destroy(self):
        _, state, mamba, host, ref = self.prepare()
        pool = mamba.MambaPoolHost(self.device_pool((0, 4)), 2, 0, layout='page_first')
        self.assertEqual(pool.temporal_buffer.numel(), 0)
        self.assertEqual(len(pool.kv_buffer), 2)
        pool.destroy()
        self.assertIsNotNone(ref())
        host.destroy()
        gc.collect()
        self.assertIsNone(ref())

    def test_failure_wrong_device_releases_owner_and_both_pools(self):
        _, state, mamba, host, ref = self.prepare()
        with self.assertRaisesRegex(RuntimeError, 'another NPU'):
            mamba.MambaPoolHost(self.device_pool(index=4), 2, 0, layout='page_first')
        gc.collect()
        self.assertIsNone(state.pool())
        self.assertIsNone(host._fl_hybrid_owner)
        self.assertIsNone(host.kv_buffer)
        self.assertIsNone(ref())

    def test_failure_after_views_and_no_strong_pending_reference(self):
        _, state, mamba, host, ref = self.prepare()
        with mock.patch.object(mamba.MambaPoolHost, '_init_write_back_staging_buffers', side_effect=RuntimeError('staging')):
            with self.assertRaisesRegex(RuntimeError, 'staging'):
                mamba.MambaPoolHost(self.device_pool(), 2, 0, layout='page_first')
        gc.collect()
        self.assertIsNone(state.pool())
        self.assertIsNone(ref())

    def test_destroy_before_mamba(self):
        _, state, _, host, ref = self.prepare()
        host.destroy()
        gc.collect()
        self.assertIsNone(state.pool())
        self.assertIsNone(ref())

    def test_logical_budget_claim_stays_decimal(self):
        module = Namespace(self.core)
        policy._install_glm(module, policy._State())
        record = module._EarlyHostArena(torch.zeros(256, dtype=torch.uint8), 3)
        record._fl_logical_budget = 200
        with self.assertRaisesRegex(RuntimeError, 'unchanged logical'):
            record.claim(201)
        self.assertIsNone(record.arena)


if __name__ == '__main__':
    # Do not run the imported SG fixture's own test suite.
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls)
                               for cls in (ArenaTests, LifecycleTests))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())
