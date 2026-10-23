# Author: liyan <yan.l@pku.edu.cn>
"""Standalone stdlib tests; no plugin package, Torch or device imports.

Run the Linux spawn tests outside model initialization and benchmark timing.
"""

import importlib.util
import multiprocessing as mp
from multiprocessing import process as mp_process
import os
from pathlib import Path
import pickle
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

_PATH = Path(__file__).resolve().parents[1] / 'sglang_fl/cpu_affinity.py'
_SPEC = importlib.util.spec_from_file_location('_fl_cpu_affinity_test', _PATH)
affinity = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = affinity
_SPEC.loader.exec_module(affinity)


def _report(writer):
    main = sorted(os.sched_getaffinity(0))
    child = []
    thread = threading.Thread(target=lambda: child.append(sorted(os.sched_getaffinity(0))))
    thread.start()
    thread.join()
    writer.send({'main': main, 'thread': child[0]})
    writer.close()


def _passthrough(target, args, kwargs):
    return target(*args, **kwargs)


class TransactionTests(unittest.TestCase):
    def test_partial_start_error_is_not_retried_and_attributes_restored(self):
        for error_type in (TypeError, ValueError, OSError):
            target = _report
            process = SimpleNamespace(_target=target, _args=('original',), _kwargs={'x': 2})
            attempts = []
            mask = {0, 1, 2, 3}

            def get_affinity(_):
                return set(mask)

            def set_affinity(_, cpus):
                mask.clear()
                mask.update(cpus)

            def fail(candidate):
                attempts.append(candidate)
                self.assertIs(candidate._target, _passthrough)
                self.assertEqual(mask, {2, 3})
                raise error_type('possibly already spawned')

            with patch.object(affinity.os, 'sched_getaffinity', get_affinity, create=True), \
                    patch.object(affinity.os, 'sched_setaffinity', set_affinity, create=True):
                with self.assertRaises(error_type):
                    affinity._replace_start(fail, process, _passthrough, {2, 3})
            self.assertEqual(len(attempts), 1)
            self.assertEqual(mask, {0, 1, 2, 3})
            self.assertIs(process._target, target)
            self.assertEqual(process._args, ('original',))
            self.assertEqual(process._kwargs, {'x': 2})

    def test_controller_failure_restores_without_affinity_or_retry(self):
        process = SimpleNamespace(_target=_report, _args=(1,), _kwargs={})
        calls = []

        def fail(candidate):
            calls.append(candidate)
            raise ValueError('controller Popen failure')

        with self.assertRaises(ValueError):
            affinity._replace_start(fail, process, _passthrough)
        self.assertEqual(len(calls), 1)
        self.assertIs(process._target, _report)
        self.assertEqual(process._args, (1,))

    def test_bootstraps_pickle_without_rebinding_native_names(self):
        for target in (affinity._controller_bootstrap, affinity._scheduler_bootstrap):
            self.assertIs(pickle.loads(pickle.dumps(target)), target)


@unittest.skipUnless(sys.platform == 'linux', 'Linux calling-TID affinity required')
class LinuxSpawnTests(unittest.TestCase):
    def test_real_spawn_inherits_two_partitions_without_changing_other_thread(self):
        original = os.sched_getaffinity(0)
        available = sorted(original)
        if len(available) < 2:
            self.skipTest('need two CPUs')
        width = min(40, len(available) // 2)
        done, ready = threading.Event(), threading.Event()
        ids = []

        def observer():
            ids.append(threading.get_native_id())
            ready.set()
            done.wait(30)

        observer_thread = threading.Thread(target=observer)
        observer_thread.start()
        self.assertTrue(ready.wait(5))
        try:
            for mask in (set(available[:width]), set(available[width:width * 2])):
                reader, writer = mp.get_context('spawn').Pipe(duplex=False)
                process = mp.get_context('spawn').Process(target=_report, args=(writer,))

                def start(candidate):
                    self.assertEqual(os.sched_getaffinity(0), mask)
                    self.assertEqual(os.sched_getaffinity(ids[0]), original)
                    return mp_process.BaseProcess.start(candidate)

                affinity._replace_start(start, process, _passthrough, mask)
                writer.close()
                self.assertFalse(hasattr(process, '_target'))
                self.assertTrue(reader.poll(20))
                result = reader.recv()
                process.join(20)
                self.assertEqual(process.exitcode, 0)
                self.assertEqual(result, {'main': sorted(mask), 'thread': sorted(mask)})
                self.assertEqual(os.sched_getaffinity(0), original)
                self.assertEqual(os.sched_getaffinity(ids[0]), original)
                reader.close()
        finally:
            done.set()
            observer_thread.join(5)
            self.assertEqual(os.sched_getaffinity(0), original)

    def test_native_pinned_process_classes_and_idempotence(self):
        if not affinity._source_contract():
            self.skipTest('not the pinned image')
        saved_start = mp_process.BaseProcess.start
        saved_context = mp.get_start_method(allow_none=True)
        try:
            mp.set_start_method('spawn', force=True)
            self.assertTrue(affinity._spawn_process(mp.Process(target=_report)))
            self.assertTrue(affinity._spawn_process(mp.get_context('spawn').Process(target=_report)))
            self.assertFalse(affinity._spawn_process(mp.get_context('fork').Process(target=_report)))
            with patch.dict(os.environ, {affinity._ENV: '1'}):
                self.assertTrue(affinity.install_cpu_affinity_hook())
                installed = mp_process.BaseProcess.start
                self.assertTrue(affinity.install_cpu_affinity_hook())
                self.assertIs(mp_process.BaseProcess.start, installed)
            self.assertTrue(affinity._topology_contract())
        finally:
            mp_process.BaseProcess.start = saved_start
            mp.set_start_method(saved_context, force=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
