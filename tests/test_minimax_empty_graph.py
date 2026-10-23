# Author: liyan <yan.l@pku.edu.cn>
"""CPU control-path checks: lazy fallback and the exact existing fence order."""

import contextlib
import gc
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch
import weakref

PATCHES = Path(__file__).resolve().parents[1] / 'sglang_fl/dispatch/backends/vendor/ascend/patches'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


empty = load('_fl_empty_cpu', PATCHES / 'minimax_m3_empty_graph.py')


def execute(self, forward_batch, pp_proxy_tensors=None):
    self.trace.append('input_copies')
    graph_key = self.key
    if forward_batch.verify:
        seq_lens = [77] * self.bs
    else:
        seq_lens = forward_batch.seq_lens.cpu().tolist() + [0] * (self.bs - self.raw_bs)
    return self.backend.replay_with_input_update(
        graph_key, seq_lens=seq_lens, attr_name='actual_seq_lengths_kv', attr_type=[]
    )


class Fixture:
    def __init__(self):
        trace = self.trace = []

        class Tensor:
            def __init__(self, values, device='npu'):
                self.values = values
                self.device = NS(type=device, index=0 if device == 'npu' else None)

            def cpu(self):
                trace.append('device_cpu')
                return Tensor(self.values, 'cpu')

            def tolist(self):
                trace.append('tolist')
                return list(self.values)

            def __getitem__(self, index):
                return Tensor(self.values[index], self.device.type)

        class Stream:
            def __init__(self, name):
                self.name = name
                self.device = 'npu:0'
                self.stream_id = 1 if name == 'replay' else 2

            def synchronize(self):
                trace.append(self.name + '_sync')
                callback = getattr(self, 'on_wait', None)
                if callback is not None:
                    self.on_wait = None
                    callback()

        class Mode:
            def __init__(self):
                self.graph_dispatch_records = []
                self.update_stream = Stream('update')

            def update_capture_record(self, cpu_update_input):
                count = len(cpu_update_input)
                if count != 1 and count != len(self.graph_dispatch_records):
                    raise RuntimeError('record count mismatch')

        class Graph:
            def __init__(self):
                self.auto_dispatch_capture = True
                self.graph_dispatch_mode = Mode()

            def update(self, cpu_update_input):
                trace.append('update')
                self.graph_dispatch_mode.update_capture_record(cpu_update_input)

            def replay(self):
                trace.append('replay')

        class Backend:
            def replay_with_input_update(self, *args, **kwargs):
                raise AssertionError('unpatched backend')

        class Sparse:
            pass

        class Hybrid:
            def __init__(self):
                self.sparse = Sparse()

        class Runner:
            def _replay_attn_backend(self):
                return self.attention

            def _get_update_attr_name(self):
                return 'actual_seq_lengths_kv'

            def _get_update_attr_type(self):
                return []

        Runner.execute = execute
        self.Tensor, self.Graph, self.Backend, self.Runner = Tensor, Graph, Backend, Runner
        self.decode = object()
        self.envs = NS(SGLANG_ENABLE_OVERLAP_PLAN_STREAM=NS(get=lambda: False))
        self.contract = empty._Contract(
            Runner, Backend, Graph, Mode, Hybrid, Sparse, self.decode,
            self.envs, lambda: False, lambda: False,
            lambda device: self.replay_stream,
        )
        self.key = Key()
        self.graph = Graph()
        self.output = object()
        self.backend = Backend()
        self.backend._graphs = {self.key: self.graph}
        self.backend._outputs = {self.key: self.output}
        self.backend._device_id = 0
        args = NS(
            disable_overlap_schedule=True, enable_two_batch_overlap=False,
            enable_single_batch_overlap=False, enable_lora=None,
            speculative_algorithm=None, cuda_graph_backend_decode='full',
            tp_size=16, dp_size=4, ep_size=1, enable_dp_attention=True,
        )
        self.runner = Runner()
        self.runner.trace, self.runner.key = trace, self.key
        self.runner.backend, self.runner.attention = self.backend, Hybrid()
        self.runner.model_runner = NS(
            server_args=args, is_draft_worker=False, spec_algorithm=NS(is_none=lambda: True),
            model_config=NS(hf_config=NS(architectures=['MiniMaxM3SparseForConditionalGeneration'])),
        )
        self.runner.capture_forward_mode = self.decode
        self.runner.captured_req_width = 1
        self.runner.is_dllm = self.runner.enable_pdmux = False
        self.runner.capture_bs = [4, 8, 12, 16]
        self.runner.bs, self.runner.raw_bs = 4, 2
        self.batch = NS(
            seq_lens=Tensor([128, 129]), seq_lens_cpu=Tensor([128, 129], 'cpu'),
            forward_mode=self.decode, verify=False,
        )
        self.runner.buffers = NS(seq_lens=self.batch.seq_lens)
        self.replay_stream = Stream('replay')

    def lazy(self):
        return self.contract.lengths(self.runner, self.batch, self.key)

    def install_serial(self):
        package = '_fl_empty_test_patches'
        parents = ['sglang', 'sglang.srt', 'sglang.srt.hardware_backend',
                   'sglang.srt.hardware_backend.npu',
                   'sglang.srt.hardware_backend.npu.graph_runner', package]
        modules = {name: ModuleType(name) for name in parents}
        for module in modules.values():
            module.__path__ = []
        backend_name = parents[-2] + '.npu_cudagraph_backend'
        modules[backend_name] = ModuleType(backend_name)
        modules[backend_name].NPUCudaGraphBackend = self.Backend
        torch = ModuleType('torch')
        torch.Tensor = self.Tensor
        torch.npu = NS(device=lambda _: contextlib.nullcontext(), current_stream=lambda: self.replay_stream)
        modules.update(torch=torch, numpy=ModuleType('numpy'))
        helper = ModuleType(package + '.minimax_m3_empty_graph')
        helper.prepare_empty_graph_update = lambda: (
            self.Runner, execute, empty._transform_execute(execute, self.contract.lengths), self.contract
        )
        modules[helper.__name__] = helper
        stages = ModuleType(package + '.graph_stages')
        stages._stage = lambda *args, **kwargs: None
        modules[stages.__name__] = stages
        self.modules = patch.dict(sys.modules, modules)
        self.modules.start()
        self.environment = patch.dict(os.environ, {
            'SGLANG_FL_ASCEND_SERIAL_GRAPH_UPDATE': '1',
            'SGLANG_FL_ASCEND_STAGE_DIAGNOSTICS': '1',
            'SGLANG_FL_ASCEND_MINIMAX_EMPTY_GRAPH_UPDATE': '1',
        })
        self.environment.start()
        self.serial = load(package + '.graph_updates', PATCHES / 'graph_updates.py')
        self.serial.patch_ascend_graph_updates()

    def close(self):
        execute.__globals__.pop(empty._HELPER, None)
        self.environment.stop()
        self.modules.stop()


class Key:
    size, stream_idx, variant_label, attention_variant = 4, None, None, None


class EmptyGraphTests(unittest.TestCase):
    def test_lazy_and_changed_records_use_original_device_payload(self):
        f = Fixture()
        value = f.lazy()
        self.assertEqual(f.trace, [])
        self.assertTrue(f.contract.resolve(value, f.backend, f.key, 'actual_seq_lengths_kv', [], None))
        f.graph.graph_dispatch_mode.graph_dispatch_records.append(object())
        self.assertFalse(f.contract.resolve(value, f.backend, f.key, 'actual_seq_lengths_kv', [], None))
        self.assertEqual(value.materialize(), [128, 129, 0, 0])
        self.assertEqual(f.trace, ['device_cpu', 'tolist'])

    def test_graph_output_identity_and_explicit_payload(self):
        for mutation in ('graph', 'output', 'explicit', 'method', 'instance', 'backend'):
            with self.subTest(mutation=mutation):
                f = Fixture()
                value = f.lazy()
                payload = None
                if mutation == 'graph':
                    f.backend._graphs[f.key] = f.Graph()
                elif mutation == 'output':
                    f.backend._outputs[f.key] = object()
                elif mutation == 'explicit':
                    payload = [{}, {}]
                elif mutation == 'instance':
                    f.graph.update = lambda *args, **kwargs: None
                elif mutation == 'backend':
                    f.runner.backend = f.Backend()
                    f.runner.backend._graphs = f.backend._graphs
                    f.runner.backend._outputs = f.backend._outputs
                else:
                    f.Graph.update = lambda *args, **kwargs: None
                self.assertFalse(f.contract.resolve(value, f.backend, f.key, 'actual_seq_lengths_kv', [], payload))

    def test_unknown_contract_falls_back(self):
        for change in ('native', 'verify', 'draft', 'records', 'mode_type', 'missing', 'overlap'):
            with self.subTest(change=change):
                f = Fixture()
                if change == 'native':
                    f.contract.native_attention = lambda: True
                elif change == 'verify':
                    f.batch.forward_mode = object()
                elif change == 'draft':
                    f.runner.model_runner.is_draft_worker = True
                elif change == 'records':
                    f.graph.graph_dispatch_mode.graph_dispatch_records = ()
                elif change == 'mode_type':
                    f.graph.graph_dispatch_mode = NS(graph_dispatch_records=[])
                elif change == 'overlap':
                    f.runner.model_runner.server_args.disable_overlap_schedule = False
                else:
                    del f.runner.model_runner.server_args.ep_size
                self.assertEqual(f.lazy(), [128, 129, 0, 0])
                self.assertEqual(f.trace, ['device_cpu', 'tolist'])

    def test_full_serial_fast_and_fallback_keep_fences(self):
        for records in ([], [NS(op_cache_entry='test_op')]):
            with self.subTest(empty=not records):
                f = Fixture()
                f.install_serial()
                try:
                    f.graph.graph_dispatch_mode.graph_dispatch_records = records
                    self.assertIs(f.runner.execute(f.batch), f.output)
                    self.assertEqual(f.trace.count('device_cpu'), bool(records))
                    self.assertEqual(f.trace.count('update'), bool(records))
                    operations = [x for x in f.trace if x in ('input_copies', 'replay_sync', 'update_sync', 'update', 'replay')]
                    expected = ['input_copies', 'replay_sync']
                    if records:
                        expected.append('update')
                    expected += ['update_sync', 'replay', 'replay_sync']
                    self.assertEqual(operations, expected)
                finally:
                    f.close()

    def test_verify_branch_and_explicit_update_error_preserved(self):
        f = Fixture()
        f.install_serial()
        try:
            f.batch.verify = True
            f.batch.forward_mode = object()
            self.assertIs(f.runner.execute(f.batch), f.output)
            self.assertNotIn('device_cpu', f.trace)
            self.assertIn('update', f.trace)
            f.batch.verify = False
            f.batch.forward_mode = f.decode
            with self.assertRaisesRegex(RuntimeError, 'record count mismatch'):
                f.backend.replay_with_input_update(
                    f.key, seq_lens=f.lazy(), attr_name='actual_seq_lengths_kv',
                    attr_type=[], cpu_update_input=[{}, {}],
                )
        finally:
            f.close()

    def test_unknown_execute_namespace_rejected_before_mutation(self):
        with patch.dict(execute.__globals__, {empty._HELPER: object()}):
            with self.assertRaisesRegex(RuntimeError, 'Unknown NPUGraphRunner.execute ABI'):
                empty._transform_execute(execute, lambda *args: None)

    def test_same_native_stream_accepts_new_python_wrapper(self):
        f = Fixture()
        value = f.lazy()
        copy = NS(device=f.replay_stream.device, stream_id=f.replay_stream.stream_id)
        f.contract.check_stream(value, copy)
        self.assertEqual(f.trace, [])

    def test_duplicate_module_alias_uses_public_constructors_module(self):
        modules = {name: ModuleType(name) for name in ('torch_npu.npu.graphs', 'torch.npu.graphs')}
        for module in modules.values():
            exec('class NPUGraph:\n    pass\n', vars(module))
        public = modules['torch_npu.npu.graphs'].NPUGraph
        self.assertIsNot(public, modules['torch.npu.graphs'].NPUGraph)
        with patch.dict(sys.modules, modules):
            self.assertIs(empty._canonical_graph_module(public), modules[public.__module__])
            modules[public.__module__].NPUGraph = modules['torch.npu.graphs'].NPUGraph
            with self.assertRaisesRegex(RuntimeError, 'Conflicting public'):
                empty._canonical_graph_module(public)

    def test_wrong_device_falls_back_before_deferring_lengths(self):
        f = Fixture()
        f.backend._device_id = 1
        self.assertEqual(f.lazy(), [128, 129, 0, 0])
        self.assertEqual(f.trace, ['device_cpu', 'tolist'])

    def test_stream_change_before_or_during_wait_rejects_replay(self):
        for during_wait in (False, True):
            with self.subTest(during_wait=during_wait):
                f = Fixture()
                f.install_serial()
                try:
                    if during_wait:
                        f.replay_stream.on_wait = lambda: setattr(f.replay_stream, 'stream_id', 9)
                        call = lambda: f.runner.execute(f.batch)
                    else:
                        value = f.lazy()
                        f.replay_stream.stream_id = 9
                        call = lambda: f.backend.replay_with_input_update(
                            f.key, seq_lens=value, attr_name='actual_seq_lengths_kv', attr_type=[])
                    with self.assertRaisesRegex(RuntimeError, 'stream changed after deferred CPU lengths'):
                        call()
                    self.assertNotIn('replay', f.trace)
                    self.assertNotIn('update', f.trace)
                finally:
                    f.close()

    def test_records_change_during_pre_replay_wait_materializes_once(self):
        f = Fixture()
        f.install_serial()
        try:
            f.replay_stream.on_wait = lambda: f.graph.graph_dispatch_mode.graph_dispatch_records.append(NS(op_cache_entry='test_op'))
            self.assertIs(f.runner.execute(f.batch), f.output)
            self.assertEqual(f.trace.count('device_cpu'), 1)
            self.assertEqual(f.trace.count('update'), 1)
            self.assertEqual(f.trace.count('replay_sync'), 2)
            self.assertEqual(f.trace.count('update_sync'), 1)
            self.assertFalse(getattr(f.backend, '_sglang_fl_empty_update_counts', {}))
        finally:
            f.close()

    def test_bindings_change_during_wait_rejects_replay(self):
        for binding in ('graph', 'output', 'mode', 'stream'):
            with self.subTest(binding=binding):
                f = Fixture()
                f.install_serial()
                try:
                    def change():
                        if binding == 'graph':
                            f.backend._graphs[f.key] = f.Graph()
                        elif binding == 'output':
                            f.backend._outputs[f.key] = object()
                        elif binding == 'mode':
                            f.graph.graph_dispatch_mode = type(f.graph.graph_dispatch_mode)()
                        else:
                            f.graph.graph_dispatch_mode.update_stream = type(f.replay_stream)('changed')
                    f.replay_stream.on_wait = change
                    with self.assertRaisesRegex(RuntimeError, 'bindings changed during serial replay wait'):
                        f.runner.execute(f.batch)
                    self.assertNotIn('replay', f.trace)
                    self.assertNotIn('update', f.trace)
                finally:
                    f.close()

    def test_statistics_do_not_retain_cleaned_graph(self):
        f = Fixture()
        f.install_serial()
        try:
            f.runner.execute(f.batch)
            reference = weakref.ref(f.graph)
            f.backend._graphs.clear()
            f.backend._outputs.clear()
            del f.graph
            gc.collect()
            self.assertIsNone(reference())
        finally:
            f.close()

    def test_reviewed_method_detects_in_memory_code_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'native.py'
            path.write_text('class Native:\n    def update(self, cpu_update_input):\n        return 123\n')
            module = load('_fl_native_method', path)
            self.assertIs(empty._reviewed_method(module, 'Native', 'update'), module.Native.update)
            module.Native.update.__code__ = module.Native.update.__code__.replace(co_consts=(None, 456))
            with self.assertRaisesRegex(RuntimeError, 'Unknown native graph method'):
                empty._reviewed_method(module, 'Native', 'update')


if __name__ == '__main__':
    unittest.main()
