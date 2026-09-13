"""Explicit two-node TP32 scope and tensor-shard graph boundaries."""
import json
from types import SimpleNamespace
import pytest


@pytest.mark.parametrize('tp,ep,nodes,wanted',[
    (16,16,1,True),(32,32,2,True),(16,16,2,False),
    (32,32,1,False),(32,16,2,False),(16,32,1,False),
    (8,8,1,False),(64,64,4,False)])
def test_normal_collective_topology(tmp_path,monkeypatch,tp,ep,nodes,wanted):
    from sglang_fl.models.glm_53_flash.normal_collective import enabled
    (tmp_path/'config.json').write_text(json.dumps({'architectures':['Glm5NextForConditionalGeneration']}))
    monkeypatch.setenv('SGLANG_FL_GLM53_NORMAL_HCCL','1')
    args=SimpleNamespace(device='npu',tp_size=tp,ep_size=ep,nnodes=nodes,pp_size=1,
        enable_dp_attention=False,enable_two_batch_overlap=False,enable_eplb=False,
        ep_num_redundant_experts=0,quantization='modelslim',model_path=str(tmp_path))
    assert enabled(args)==wanted


@pytest.mark.parametrize('requests,tokens,wanted',[
    (1,4,(32,8)),(17,4,(32,24)),(33,4,(64,40)),
    (33,5,(64,64)),(64,4,(64,64))])
def test_tp32_graph_counts(requests,tokens,wanted):
    from sglang_fl.models.glm_53_flash.mtp_draft_graph import capture_counts
    assert capture_counts(requests,32,tokens)==wanted
