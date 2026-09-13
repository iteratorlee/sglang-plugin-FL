"""Small graph selection preserves opt-in compilation and model scope."""

from types import SimpleNamespace
import pytest


def runner(compile_enabled=False, pool_size=1):
    return SimpleNamespace(
        device='npu',
        model_config=SimpleNamespace(hf_config=SimpleNamespace(
            architectures=['Glm5NextForConditionalGeneration'])),
        req_to_token_pool=SimpleNamespace(size=pool_size),
        server_args=SimpleNamespace(enable_dp_attention=False,
            speculative_algorithm=None, enable_two_batch_overlap=False,
            cuda_graph_bs=[1, 2, 4], tp_size=16,
            enable_torch_compile=compile_enabled, torch_compile_max_bs=2))


@pytest.mark.parametrize('compile_enabled', [False, True])
@pytest.mark.parametrize('pool_size', [1, 4])
def test_small_graph_capture_does_not_enable_compile(compile_enabled, pool_size):
    from sglang_fl.models.glm_53_flash.small_graph import patch_small_graphs
    from sglang.srt.model_executor import cuda_graph_runner
    patch_small_graphs()
    sizes, compiled = cuda_graph_runner.get_batch_sizes_to_capture(runner(compile_enabled, pool_size))
    assert sizes == ([1] if pool_size == 1 else [1, 2, 4])
    assert compiled == ([s for s in sizes if s <= 2] if compile_enabled else [])


@pytest.mark.parametrize('unsupported', ['device', 'architecture', 'dp', 'speculative'])
def test_small_graph_scope(unsupported):
    from sglang_fl.models.glm_53_flash.small_graph import _enabled
    r = runner()
    if unsupported == 'device': r.device = 'cuda'
    if unsupported == 'architecture': r.model_config.hf_config.architectures = ['Other']
    if unsupported == 'dp': r.server_args.enable_dp_attention = True
    if unsupported == 'speculative': r.server_args.speculative_algorithm = 'EAGLE'
    assert not _enabled(r)
