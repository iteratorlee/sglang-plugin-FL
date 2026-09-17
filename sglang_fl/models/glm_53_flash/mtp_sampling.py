"""Preserve target sampling when v0.5.11 has no NPU tree-sampling kernel.

Draw independently from the target distribution at each verified tree node,
then reuse greedy tree *matching* with those draws. A draft edge is accepted
only when it equals the target draw; the first unmatched draw is the bonus
token. This is exact target-only speculative sampling, with no draft-probability
approximation. The existing verifier still owns EOS, KV and KDA commits.
"""

from contextvars import ContextVar
from functools import wraps
import logging

import torch

logger = logging.getLogger(__name__)
_CONTEXT = ContextVar("glm53_mtp_target_sampling", default=None)
_PATCHED = False


def sample_logits_npu(logits, top_ks, top_ps, min_ps, need_min_p):
    from sglang.srt.layers.sampler import top_k_top_p_min_p_sampling_from_logits_ascend
    if not need_min_p:
        return top_k_top_p_min_p_sampling_from_logits_ascend(
            logits, top_ks, top_ps, min_ps, False,
        )
    # v0.5.11's fused Ascend min-p branch multiplies the (values, indices)
    # tuple returned by max(). Keep min-p on the equivalent tensor fallback.
    probabilities, indices = logits.softmax(-1).sort(-1, descending=True)
    ranks = torch.arange(logits.shape[-1], device=logits.device)[None, :]
    probabilities.masked_fill_(ranks >= top_ks[:, None], 0)
    prefix = probabilities.cumsum(-1) - probabilities
    probabilities.masked_fill_(prefix > top_ps[:, None], 0)
    probabilities.masked_fill_(probabilities < probabilities[:, :1] * min_ps[:, None], 0)
    choice = torch.multinomial(probabilities, 1)
    return indices.gather(1, choice).flatten()


def sample_target_nodes(logits, sampling_info, draft_token_num, sample_logits):
    """Use the same temperature/top-k/top-p/min-p law as the NPU sampler.

    The original verifier has already applied penalties, custom processors and
    grammar masks. Never overwrite its logits (they are also used for logprobs),
    or the sampling metadata (the Ascend sampler can modify its top-k argument).
    """
    temperatures = sampling_info.temperatures.repeat_interleave(draft_token_num, 0)
    top_ks = sampling_info.top_ks.repeat_interleave(draft_token_num, 0)
    top_ps = sampling_info.top_ps.repeat_interleave(draft_token_num, 0)
    min_ps = sampling_info.min_ps.repeat_interleave(draft_token_num, 0)
    if logits.shape[0] != temperatures.shape[0]:
        raise ValueError("GLM MTP sampling metadata does not match target logits")
    draws = sample_logits(
        logits / temperatures, top_ks, top_ps, min_ps,
        sampling_info.need_min_p_sampling,
    )
    return draws.to(torch.int32).reshape(-1, draft_token_num)


class _HandledFallback(logging.Filter):
    def filter(self, record):
        # Only suppress the old warning inside a request actually handled here.
        # Other models/devices and other warnings retain their original behavior.
        return not (
            _CONTEXT.get() is not None
            and record.getMessage().startswith("Tree speculative sampling kernel unavailable")
        )


def patch_mtp_sampling():
    global _PATCHED
    if _PATCHED:
        return
    from sglang.srt.speculative import eagle_info
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.layers.dp_attention import (
        get_attention_tp_group, is_dp_attention_enabled,
    )

    original_verify = eagle_info.EagleVerifyInput.verify
    original_match = eagle_info.verify_tree_greedy_func

    @wraps(original_verify)
    def verify(self, batch, logits_output, *args, **kwargs):
        architectures = getattr(batch.model_config.hf_config, "architectures", ()) or ()
        if (
            eagle_info.TREE_SPEC_KERNEL_AVAILABLE
            or "Glm5NextForConditionalGeneration" not in architectures
            or batch.forward_mode.is_idle()
            or batch.sampling_info.is_all_greedy
            or logits_output.next_token_logits.device.type != "npu"
        ):
            return original_verify(self, batch, logits_output, *args, **kwargs)
        if self.topk != 1:
            raise NotImplementedError("GLM NPU stochastic MTP currently requires topk=1")
        if self.retrieve_index.shape[0] != len(batch.sampling_info):
            raise ValueError("GLM stochastic MTP requires unfiltered sampling metadata")
        token = _CONTEXT.set((logits_output, batch.sampling_info, self.draft_token_num))
        try:
            return original_verify(self, batch, logits_output, *args, **kwargs)
        finally:
            _CONTEXT.reset(token)

    @wraps(original_match)
    def match(*args, **kwargs):
        context = _CONTEXT.get()
        if context is not None:
            output, sampling_info, draft_token_num = context
            target_predict = sample_target_nodes(
                output.next_token_logits, sampling_info, draft_token_num,
                sample_logits_npu,
            )
            # Every TP rank must traverse and commit exactly the same prefix.
            group = get_attention_tp_group() if is_dp_attention_enabled() else get_tp_group()
            if group.world_size > 1:
                group.broadcast(target_predict, src=0)
            kwargs["target_predict"] = target_predict
        return original_match(*args, **kwargs)

    eagle_info.EagleVerifyInput.verify = verify
    eagle_info.verify_tree_greedy_func = match
    eagle_info.logger.addFilter(_HandledFallback())
    _PATCHED = True
    logger.info("GLM NPU MTP target sampling enabled; greedy path unchanged")
