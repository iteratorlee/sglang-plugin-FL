# Copyright 2026 FlagOS Contributors
# Author: liyan <yan.l@pku.edu.cn>
# SPDX-License-Identifier: Apache-2.0
"""Opt-in scheduler CPU partitions for the reviewed single-host Ascend image.

Only the thread calling native spawn is temporarily bound. Linux exec and new
threads inherit that mask; the controller's other threads and memory policy are
untouched. Unknown sources, topology, targets or arguments retain native start.
This deliberately does not implement a general NUMA or device-locality policy.
"""

import hashlib
import inspect
import json
import logging
import multiprocessing as mp
from multiprocessing import context as mp_context
from multiprocessing import process as mp_process
import os
from pathlib import Path
import sys
import threading
from types import CodeType, FunctionType

logger = logging.getLogger(__name__)
_ENV = 'SGLANG_FL_ASCEND_SCHEDULER_CPU_PARTITION'
_ALL_CPUS = frozenset(range(640))
_SG = '/sgl-workspace/sglang/python/sglang/'
_MP = '/usr/local/python3.11.15/lib/python3.11/multiprocessing/'
_SOURCES = {
    _MP + 'process.py': '67f0c2a7a3a83c92dd024705bac18619a2e123c9df77c414beb81035ea4a0e18',
    _MP + 'context.py': 'bd8628d1e140d933d40ed14b51009820c1b0b9b5b6c71971df5fc761da65bd08',
    _MP + 'popen_spawn_posix.py': '97b5d25aa479516894489877e6a7921252ee35f51e118c2f1f91f32919e7caa8',
    _MP + 'spawn.py': 'ebf9fa40eb622384c37690d8c78e7208744df031155ab4ceedab0fc791a1669b',
    _MP + 'util.py': '6752c4515ec69f82e9df64e017da490c3754e51d818c270ea1ad2d64e09268be',
    _SG + 'srt/entrypoints/engine.py': '9e5833f8585ce7abfa2fae09f021b532fcee2de2f53d1294f06ca9b4f29a8d1a',
    _SG + 'srt/managers/data_parallel_controller.py': '0aa037b0de1b1b3c1473212ac3668877331ae8acd17e43c626b2656f488c0959',
    _SG + 'srt/managers/scheduler.py': '0984d8c757bbebf27982aad23413261b05b7682d7371ad733089d81536001044',
    _SG + 'srt/utils/numa_utils.py': '6a57fef1f2ad88e2499f93a339f7ba678034e1fd8e349a91ca99c3b94e08ab98',
}
_TARGETS = {
    'controller': ('sglang.srt.managers.data_parallel_controller',
                   'run_data_parallel_controller_process',
                   ('server_args', 'port_args', 'pipe_writer', 'run_scheduler_process_func')),
    'scheduler': ('sglang.srt.managers.scheduler', 'run_scheduler_process',
                  ('server_args', 'port_args', 'gpu_id', 'tp_rank', 'attn_cp_rank',
                   'moe_dp_rank', 'moe_ep_rank', 'pp_rank', 'dp_rank', 'pipe_writer',
                   'display_tp_rank', 'display_dp_rank', 'display_moe_ep_rank')),
}
_source_pid = None
_source_ok = False
_topology_pid = None
_topology_ok = False
_compiled = {}
_installed_start = None
_installed_pid = None


def _event(stage, **fields):
    logger.warning('FL_NPU_CPU_AFFINITY %s', json.dumps({
        'stage': stage, 'pid': os.getpid(), 'tid': threading.get_native_id(), **fields,
    }, sort_keys=True))


def _reject(reason, **fields):
    _event('rejected', reason=reason, **fields)
    return False


def _source_contract():
    global _source_pid, _source_ok
    if _source_pid == os.getpid():
        return _source_ok
    _source_pid, _source_ok = os.getpid(), False
    if sys.platform != 'linux' or sys.version_info[:3] != (3, 11, 15):
        return _reject('python_platform', platform=sys.platform, version=list(sys.version_info[:3]))
    try:
        for path, digest in _SOURCES.items():
            actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if actual != digest:
                return _reject('source_sha256', path=path, expected=digest, actual=actual)
        _source_ok = True
    except OSError as error:
        return _reject('source_read', path=error.filename, errno=error.errno)
    return _source_ok


def _code_objects(code):
    yield code
    for child in code.co_consts:
        if isinstance(child, CodeType):
            yield from _code_objects(child)


def _native_function(function, module_name, qualified_name, path):
    """Check loaded identity as well as the source on disk, without importing SG."""
    module = sys.modules.get(module_name)
    if (type(function) is not FunctionType or module is None
            or function.__module__ != module_name or function.__qualname__ != qualified_name
            or function.__globals__ is not vars(module)):
        return False
    try:
        if (Path(module.__file__).resolve() != Path(path).resolve()
                or Path(module.__spec__.origin).resolve() != Path(path).resolve()):
            return False
        if path not in _compiled:
            # context.py defines SpawnProcess in both the POSIX and Windows
            # branches. Preserve both pinned code objects, not the last branch.
            grouped = {}
            for code in _code_objects(compile(Path(path).read_text(), path, 'exec',
                    dont_inherit=True, optimize=sys.flags.optimize)):
                grouped.setdefault(code.co_qualname, []).append(code)
            _compiled[path] = grouped
        return function.__code__ in _compiled[path][qualified_name]
    except (OSError, KeyError, AttributeError, TypeError, SyntaxError):
        return False


def _topology_contract():
    global _topology_pid, _topology_ok
    if _topology_pid == os.getpid():
        return _topology_ok
    _topology_pid, _topology_ok = os.getpid(), False
    try:
        root = Path('/sys/devices/system/cpu')
        if (root / 'online').read_text().strip() != '0-639':
            return False
        physical = set()
        for cpu in range(640):
            topology = root / f'cpu{cpu}/topology'
            first = cpu // 2 * 2
            if (topology / 'thread_siblings_list').read_text().strip() != f'{first}-{first + 1}':
                return False
            physical.add(((topology / 'physical_package_id').read_text().strip(),
                          (topology / 'core_id').read_text().strip()))
        _topology_ok = len(physical) == 320
    except OSError:
        pass
    return _topology_ok


def _native_target(target, kind):
    module_name, name, parameters = _TARGETS[kind]
    path = _SG + module_name.removeprefix('sglang.').replace('.', '/') + '.py'
    module = sys.modules.get(module_name)
    return (module is not None and getattr(module, name, None) is target
            and _native_function(target, module_name, name, path)
            and tuple(inspect.signature(target).parameters) == parameters)


def _server_contract(args):
    expected = {
        'device': 'npu', 'tp_size': 16, 'dp_size': 4, 'ep_size': 1, 'pp_size': 1,
        'attn_cp_size': 1, 'dcp_size': 1, 'moe_dp_size': 1, 'dwdp_size': 1,
        'nnodes': 1, 'node_rank': 0, 'base_gpu_id': 0, 'gpu_id_step': 1,
        'enable_dp_attention': True, 'numa_node': None, 'elastic_ep_backend': None,
        'enable_elastic_expert_backup': False, 'elastic_ep_initial_size': None,
        'elastic_ep_rejoin': False,
    }
    module = sys.modules.get('sglang.srt.server_args')
    if module is None or type(args) is not getattr(module, 'ServerArgs', None):
        return _reject('server_args_type')
    for key, value in expected.items():
        if not hasattr(args, key) or getattr(args, key) != value:
            return _reject('server_args_field', field=key, expected=value,
                           actual=getattr(args, key, '<missing>'))
    for key in ('SGLANG_SET_CPU_AFFINITY', 'SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS'):
        if os.environ.get(key, '0').lower() not in ('0', 'false', ''):
            return _reject('conflicting_affinity_env', field=key)
    if os.environ.get('SGLANG_NUMA_BIND_V2', '1').lower() not in ('1', 'true'):
        return _reject('numa_bind_v2_env')
    for key in ('ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_VISIBLE_DEVICES'):
        if os.environ.get(key, '') not in ('', ','.join(map(str, range(16)))):
            return _reject('device_order_env', field=key, actual=os.environ[key])
    return True


def _spawn_process(process):
    if 'run' in vars(process) or '_Popen' in vars(process):
        return False
    if not _native_function(type(process).run, 'multiprocessing.process',
                            'BaseProcess.run', _MP + 'process.py'):
        return False
    spawn_popen = mp_context.SpawnProcess._Popen
    if (not _native_function(spawn_popen, 'multiprocessing.context',
                             'SpawnProcess._Popen', _MP + 'context.py')
            or spawn_popen.__code__.co_names != ('popen_spawn_posix', 'Popen')):
        return False
    if type(process) is mp_context.Process:
        return (mp.get_start_method(allow_none=True) == 'spawn'
                and _native_function(type(process)._Popen, 'multiprocessing.context',
                                     'Process._Popen', _MP + 'context.py')
                )
    return (type(process) is mp_context.SpawnProcess
            and _native_function(type(process)._Popen, 'multiprocessing.context',
                                 'SpawnProcess._Popen', _MP + 'context.py'))


def _bound_start(native_start, process, cpus):
    """Transactional calling-TID binding; never retry a possibly partial spawn."""
    before = os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0, cpus)
        if os.sched_getaffinity(0) != set(cpus):
            raise RuntimeError('CPU partition was clipped by the effective cpuset')
        return native_start(process)
    finally:
        os.sched_setaffinity(0, before)
        restored = os.sched_getaffinity(0)
        if restored != before:
            raise RuntimeError('Calling thread CPU affinity was not restored')
        _event('parent_restored', cpus=sorted(restored))


def _replace_start(native_start, process, bootstrap, cpus=None):
    saved = process._target, process._args, process._kwargs
    process._target, process._args, process._kwargs = bootstrap, saved, {}
    try:
        if cpus is None:
            return native_start(process)
        return _bound_start(native_start, process, cpus)
    except BaseException:
        # A failed Popen may already have created a child: restore, propagate,
        # and let native SG handle failure. Never start the process a second time.
        if hasattr(process, '_target'):
            process._target, process._args, process._kwargs = saved
        raise


def _controller_bootstrap(target, args, kwargs):
    if (not _source_contract() or os.sched_getaffinity(0) != _ALL_CPUS
            or not _native_target(target, 'controller')
            or not install_cpu_affinity_hook()):
        raise RuntimeError('CPU partition controller bootstrap contract failed')
    _event('controller_bootstrap', cpus=sorted(os.sched_getaffinity(0)))
    # The native controller never calls load_plugins(). Do not enable dispatch,
    # FlagGems or any unrelated hooks in a process that previously lacked them.
    return target(*args, **kwargs)


def _scheduler_bootstrap(target, args, kwargs):
    if not _source_contract() or not _native_target(target, 'scheduler'):
        raise RuntimeError('CPU partition scheduler bootstrap target changed')
    bound = inspect.signature(target).bind(*args, **kwargs)
    rank = bound.arguments['tp_rank']
    cpus = set(range(rank * 40, (rank + 1) * 40))
    if os.sched_getaffinity(0) != cpus:
        raise RuntimeError('Scheduler did not inherit its CPU partition')
    _event('scheduler_bootstrap', rank=rank, cpus=sorted(cpus))
    # This native target itself calls load_plugins() before constructing Scheduler.
    return target(*args, **kwargs)


def install_cpu_affinity_hook():
    """Install early, but resolve the native start method only when starting SG."""
    global _installed_start, _installed_pid
    if os.environ.get(_ENV) != '1':
        return False
    current = mp_process.BaseProcess.start
    if current is _installed_start and _installed_pid == os.getpid():
        return True
    if (not _source_contract() or not _native_function(current, 'multiprocessing.process',
            'BaseProcess.start', _MP + 'process.py')):
        _event('rejected', reason='source_contract')
        return False

    def start(process):
        target = getattr(process, '_target', None)
        kind = next((kind for kind, (module, name, _) in _TARGETS.items()
                     if getattr(target, '__module__', None) == module
                     and getattr(target, '__name__', None) == name), None)
        if kind is None:
            return current(process)
        for reason, check in (
                ('process_class_or_start_method', lambda: _spawn_process(process)),
                ('native_target', lambda: _native_target(target, kind)),
                ('cpu_topology', _topology_contract),
                ('parent_affinity', lambda: os.sched_getaffinity(0) == _ALL_CPUS)):
            if not check():
                _reject(reason, kind=kind)
                return current(process)
        try:
            bound = inspect.signature(target).bind(*process._args, **process._kwargs)
            bound.apply_defaults()
            if not _server_contract(bound.arguments['server_args']):
                raise ValueError('server_contract')
            if kind == 'controller':
                if not _native_target(bound.arguments['run_scheduler_process_func'], 'scheduler'):
                    raise ValueError('custom_scheduler_target')
            else:
                values = bound.arguments
                rank = values['tp_rank']
                if (type(rank) is not int or not 0 <= rank < 16
                        or values['gpu_id'] != rank or values['dp_rank'] != rank // 4
                        or values['display_tp_rank'] != rank
                        or values['display_dp_rank'] != rank // 4
                        or values['display_moe_ep_rank'] != 0
                        or any(values[key] != 0 for key in
                               ('attn_cp_rank', 'moe_dp_rank', 'moe_ep_rank', 'pp_rank'))):
                    raise ValueError('rank_contract')
        except (TypeError, ValueError) as error:
            _event('rejected', reason=str(error), kind=kind)
            return current(process)
        # Keep native start outside the validation except: any Popen exception
        # must propagate without retry, including TypeError and ValueError.
        if kind == 'controller':
            return _replace_start(current, process, _controller_bootstrap)
        return _replace_start(current, process, _scheduler_bootstrap,
                              set(range(rank * 40, (rank + 1) * 40)))

    start._sglang_fl_cpu_partition = True
    start.__wrapped__ = current
    mp_process.BaseProcess.start = start
    _installed_start, _installed_pid = start, os.getpid()
    _event('installed')
    return True
