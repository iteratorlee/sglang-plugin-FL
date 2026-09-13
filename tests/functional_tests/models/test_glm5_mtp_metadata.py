"""TP token padding must not invent extra MTP request sequences."""
from types import SimpleNamespace
import pytest
import torch


@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize('batch,steps', [(1,2),(2,2),(1,5),(2,5)])
def test_verify_metadata_excludes_tp_padding(batch,steps,padded):
    from sglang_fl.models.glm_53_flash.ascend_kda import AscendKDAAttnBackend
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    backend=AscendKDAAttnBackend.__new__(AscendKDAAttnBackend)
    backend.device='cpu'
    backend.topk=1
    backend.req_to_token_pool=SimpleNamespace(get_mamba_indices=lambda indices: indices.int()+1)
    fb=SimpleNamespace(batch_size=batch,req_pool_indices=torch.arange(1,batch+1),
        forward_mode=ForwardMode.TARGET_VERIFY,input_ids=torch.zeros(16,dtype=torch.int64),
        spec_info=SimpleNamespace(draft_token_num=steps),mamba_track_mask=None)
    if padded:
        fb._original_batch_size = batch
        fb.batch_size = 8
        fb.req_pool_indices = torch.cat((fb.req_pool_indices, torch.zeros(8-batch, dtype=torch.int64)))
    metadata=backend._forward_metadata(fb)
    assert metadata.query_start_loc.tolist()==list(range(0,(batch+1)*steps,steps))
    assert metadata.mamba_cache_indices.tolist()==list(range(2,batch+2))
