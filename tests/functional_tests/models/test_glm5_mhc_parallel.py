"""Communication-boundary tests with distinguishable DP/token/rank data."""

from types import SimpleNamespace
import pytest
import torch


@pytest.mark.parametrize("tp_size,dp_size", [(16, 1), (4, 4), (4, 8)])
@pytest.mark.parametrize("scattered", [True, False])
@pytest.mark.parametrize("tokens", [0, 16])
def test_mhc_parallel_boundaries(monkeypatch, tp_size, dp_size, scattered, tokens):
    from sglang_fl.models.glm_53_flash import mhc_communicator as mod

    hidden = 8
    local = torch.arange(tokens * hidden, dtype=torch.float32).reshape(tokens, hidden)
    residual = torch.zeros(tokens, 4 * hidden)
    local_mlp = local * 3 + 7
    global_mlp = torch.cat([local_mlp + 1000 * r for r in range(dp_size)], dim=0)
    for rank in range(tp_size):
        calls = []
        monkeypatch.setattr(mod, "get_attention_tp_size", lambda: tp_size)
        monkeypatch.setattr(mod, "get_attention_tp_rank", lambda: rank)
        monkeypatch.setattr(mod, "get_attention_dp_size", lambda: dp_size)

        def all_reduce(x):
            if "attention_reduce" not in calls:
                calls.append("attention_reduce")
                return x * tp_size
            calls.append("tp_gather")
            chunk = tokens // tp_size
            shard = torch.zeros_like(local_mlp)
            shard[rank * chunk : (rank + 1) * chunk] = local_mlp[
                rank * chunk : (rank + 1) * chunk
            ]
            torch.testing.assert_close(x, shard)
            return local_mlp.clone()

        monkeypatch.setattr(
            mod, "attention_tensor_model_parallel_all_reduce", all_reduce
        )
        monkeypatch.setattr(
            mod, "get_global_dp_buffer", lambda: torch.empty(tokens * dp_size, hidden)
        )

        def gather_dp(out, x, batch, is_partial):
            assert is_partial is False
            calls.append("dp_gather")
            out.copy_(torch.cat([x + 1000 * r for r in range(dp_size)]))

        def scatter_dp(out, x, batch):
            calls.append("dp_scatter")
            out.copy_(x[:tokens])

        def gather_tp(out, x):
            calls.append("tp_gather")
            chunk = tokens // tp_size
            torch.testing.assert_close(x, local_mlp[rank * chunk : (rank + 1) * chunk])
            out.copy_(local_mlp)

        monkeypatch.setattr(mod, "_dp_gather_via_all_reduce", gather_dp)
        monkeypatch.setattr(mod, "dp_scatter", scatter_dp)
        mode = mod.ScatterMode.SCATTERED if scattered else mod.ScatterMode.FULL
        comm = mod.MHCLayerCommunicator(
            SimpleNamespace(mlp_mode=mode),
            None,
            None,
            is_first_layer=False,
            hc_mult=4,
            hc_attn_pre=None,
            hc_ffn_pre=None,
            hc_post=None,
        )

        # Isolate communication from already-tested mHC arithmetic, while
        # checking that the complete local residual survives token sharding.
        class State:
            hc_mult = 4
            h_res = object()
            h_post = object()

            def combine(self, x, r):
                assert r is residual
                return x

            def split(self, x, pre, norm):
                torch.testing.assert_close(x, local * tp_size)
                return local_mlp.clone(), residual

            hc_ffn_pre = None

        comm.mhc = State()
        mlp_input, r = comm.prepare_mlp(local.clone(), residual, SimpleNamespace())
        assert r is residual
        if scattered:
            chunk = tokens // tp_size
            torch.testing.assert_close(
                mlp_input, local_mlp[rank * chunk : (rank + 1) * chunk]
            )
            output, _ = comm.postprocess_layer(mlp_input, residual, SimpleNamespace())
        else:
            torch.testing.assert_close(
                mlp_input, global_mlp if dp_size > 1 else local_mlp
            )
            output, _ = comm.postprocess_layer(mlp_input, residual, SimpleNamespace())
        torch.testing.assert_close(output, local_mlp)
        assert comm.mhc.h_res is comm.mhc.h_post is None
        assert ("dp_gather" in calls) == (not scattered and dp_size > 1)
        assert ("tp_gather" in calls) == (scattered and tokens > 0)
