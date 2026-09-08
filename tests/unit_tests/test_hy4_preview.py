from types import SimpleNamespace

import pytest

from sglang_fl.models.hy4_preview.bootstrap import (
    HYV4Config,
    HYV4_SAFE_GRAPH_BATCH_SIZE,
    apply_sglang_patches,
)


def test_hy4_native_dsa_config_preserves_checkpoint_limits():
    quantization_config = {
        "quant_method": "compressed-tensors",
        "ignore": [],
        "config_groups": {"w8a8": {"targets": []}},
    }
    config = HYV4Config(
        architectures=["HYV4ForCausalLM"],
        index_topk=2048,
        max_position_embeddings=1_048_576,
        layer_types=["deepseek_sparse_attention"] * 2,
        indexer_types=["full", "shared"],
        quantization_config=quantization_config,
    )

    assert config.architectures[:2] == [
        "HYV4ForCausalLM",
        "DeepseekV3ForCausalLM",
    ]
    assert config.index_topk == 2048
    assert config.max_position_embeddings == 1_048_576
    assert config.index_topk_pattern == ["F", "S"]
    assert not any("gate_up_proj" in item for item in config.quantization_config["ignore"])
    assert "re:.*\\.indexer\\..*$" in config.quantization_config["ignore"]


def test_hy4_service_accepts_native_dsa_long_context(monkeypatch):
    apply_sglang_patches()
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt import server_args as server_args_module

    config = HYV4Config(
        architectures=["HYV4ForCausalLM"],
        index_topk=2048,
        max_position_embeddings=1_048_576,
    )
    fake_model_config = SimpleNamespace(
        hf_config=config,
        hf_text_config=config,
        is_draft_model=False,
    )
    monkeypatch.setattr(
        server_args_module,
        "get_global_server_args",
        lambda: SimpleNamespace(disable_cuda_graph=True),
    )
    ModelConfig._derive_context_length(fake_model_config, 32_768)
    assert fake_model_config.context_len == 32_768


def test_hy4_service_rejects_unsafe_graph_capture_batch(monkeypatch):
    apply_sglang_patches()
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt import server_args as server_args_module

    fake_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["HYV4ForCausalLM"])
    )
    unsafe_server_args = SimpleNamespace(
        disable_cuda_graph=False,
        cuda_graph_bs=[1, HYV4_SAFE_GRAPH_BATCH_SIZE + 1],
        cuda_graph_max_bs=HYV4_SAFE_GRAPH_BATCH_SIZE + 1,
        max_running_requests=HYV4_SAFE_GRAPH_BATCH_SIZE,
    )
    monkeypatch.setattr(
        server_args_module,
        "get_global_server_args",
        lambda: unsafe_server_args,
    )
    with pytest.raises(ValueError, match="NPUGraph is validated only"):
        ModelConfig._derive_context_length(fake_model_config, 32_768)


def test_hy4_service_accepts_graph_capture_batch_at_safe_limit(monkeypatch):
    apply_sglang_patches()
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt import server_args as server_args_module

    config = HYV4Config(
        architectures=["HYV4ForCausalLM"],
        index_topk=2048,
        max_position_embeddings=1_048_576,
    )
    fake_model_config = SimpleNamespace(
        hf_config=config,
        hf_text_config=config,
        is_draft_model=False,
    )
    safe_server_args = SimpleNamespace(
        disable_cuda_graph=False,
        cuda_graph_bs=[1, HYV4_SAFE_GRAPH_BATCH_SIZE],
        cuda_graph_max_bs=HYV4_SAFE_GRAPH_BATCH_SIZE,
        max_running_requests=HYV4_SAFE_GRAPH_BATCH_SIZE,
    )
    monkeypatch.setattr(
        server_args_module,
        "get_global_server_args",
        lambda: safe_server_args,
    )
    ModelConfig._derive_context_length(fake_model_config, 32_768)
    assert fake_model_config.context_len == 32_768


@pytest.mark.parametrize(
    "unsafe_max_running",
    [None, HYV4_SAFE_GRAPH_BATCH_SIZE + 1],
)
def test_hy4_service_rejects_unsafe_graph_concurrency(
    monkeypatch, unsafe_max_running
):
    apply_sglang_patches()
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt import server_args as server_args_module

    fake_model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["HYV4ForCausalLM"])
    )
    unsafe_server_args = SimpleNamespace(
        disable_cuda_graph=False,
        cuda_graph_bs=[1, HYV4_SAFE_GRAPH_BATCH_SIZE],
        cuda_graph_max_bs=HYV4_SAFE_GRAPH_BATCH_SIZE,
        max_running_requests=unsafe_max_running,
    )
    monkeypatch.setattr(
        server_args_module,
        "get_global_server_args",
        lambda: unsafe_server_args,
    )
    with pytest.raises(ValueError, match="max_running_requests"):
        ModelConfig._derive_context_length(fake_model_config, 32_768)
