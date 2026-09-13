from types import SimpleNamespace
import pytest


def args(**changes):
    config = dict(tp_size=16,ep_size=16,moe_a2a_backend='deepep',deepep_mode='auto',
        max_running_requests=1,enable_two_batch_overlap=False,dp_size=1,
        enable_dp_attention=False,cuda_graph_bs=[1],speculative_num_draft_tokens=None)
    config.update(changes)
    return SimpleNamespace(**config)


@pytest.mark.parametrize('batch', [1, 2])
@pytest.mark.parametrize('dp', [1, 4])
@pytest.mark.parametrize('draft_tokens,expected', [(None,1),(2,2),(3,4),(5,16)])
def test_capacity_covers_padded_requests(batch,dp,draft_tokens,expected):
    from sglang_fl.models.glm_53_flash.deepep_tuning import small_batch_capacity
    running = max(batch,dp)
    value=small_batch_capacity(args(max_running_requests=running,dp_size=dp,
        enable_dp_attention=dp>1,cuda_graph_bs=[1,16//dp],
        speculative_num_draft_tokens=draft_tokens),16//dp)
    assert value==expected


@pytest.mark.parametrize('changes', [dict(tp_size=32,ep_size=32),dict(deepep_mode='low_latency'),
    dict(max_running_requests=None),dict(max_running_requests=32),dict(enable_two_batch_overlap=True)])
def test_untested_config_retains_default(changes):
    from sglang_fl.models.glm_53_flash.deepep_tuning import small_batch_capacity
    assert small_batch_capacity(args(**changes),16)==128


def test_explicit_capacity_wins(monkeypatch):
    from sglang_fl.models.glm_53_flash.deepep_tuning import configure_small_batch_capacity
    name='SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK'
    monkeypatch.setenv(name,'128')
    configure_small_batch_capacity(args(),16)
    import os
    assert os.environ[name]=='128'
