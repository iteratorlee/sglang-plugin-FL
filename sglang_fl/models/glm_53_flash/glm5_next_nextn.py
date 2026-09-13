"""GLM-5.3's checkpoint MTP layer, retaining ModelSlim INT8 expert/linear GEMMs."""

from copy import deepcopy
import logging
import torch
from torch import nn

from sglang.srt.distributed import get_pp_group
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import BumpAllocator

from .checkpoint import canonical_weight_name
from .glm5_next import Glm5NextDecoderLayer, Glm5NextForConditionalGeneration, get_attn_tp_context, get_parallel
from .modelslim import GlmModelSlimConfig

logger = logging.getLogger(__name__)


class Glm53NextNModel(nn.Module):
    def __init__(self, config, quant_config):
        super().__init__()
        self.start_layer, self.end_layer = 0, 1
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size,
            use_attn_tp_group=is_dp_attention_enabled(), prefix='model.embed_tokens')
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(2*config.hidden_size, config.hidden_size, bias=False)
        self.rot = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.decoder = Glm5NextDecoderLayer(config, 0, quant_config=quant_config,
            is_nextn=True, prefix=f'model.layers.{config.num_hidden_layers}')
        self.shared_head = nn.Module()
        self.shared_head.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions, forward_batch):
        hidden = self.embed_tokens(input_ids)
        if hidden.shape[0]:
            hidden = hidden.masked_fill(positions.eq(0).unsqueeze(-1), 0)
            previous = forward_batch.spec_info.hidden_states
            if previous.shape != hidden.shape:
                raise ValueError('GLM MTP requires contracted, normalized target hidden states')
            hidden = self.eh_proj(torch.cat((self.enorm(hidden),
                self.hnorm(self.rot(previous))), -1))
        zeros = BumpAllocator(buffer_size=2, dtype=torch.float32, device=input_ids.device)
        with get_global_expert_distribution_recorder().disable_this_region():
            hidden, residual, _ = self.decoder(positions, hidden, forward_batch, None, zeros)
        if not forward_batch.forward_mode.is_idle():
            if residual is not None:
                hidden, _ = self.shared_head.norm(hidden, residual)
            else:
                hidden = self.shared_head.norm(hidden)
        return hidden


class Glm5NextForConditionalGenerationNextN(Glm5NextForConditionalGeneration):
    def __init__(self, config, quant_config=None, prefix=''):
        nn.Module.__init__(self)
        self.config = deepcopy(getattr(config, 'text_config', config))
        self.config.num_hidden_layers = self.config.glm53_nextn_layer_id
        # Unlike the 45 target blocks, the exported MTP block has ordinary
        # residual connections and contains no hc_attn/ffn parameters.
        self.config.mhc = False
        if quant_config is None or quant_config.get_name() != 'modelslim':
            raise ValueError('GLM-5.3 MTP requires ModelSlim W8A8')
        self.quant_config = GlmModelSlimConfig(quant_config)
        self.quant_config.glm53_is_mtp = True
        if not self.quant_config.quant_description.get('is_rot_used'):
            raise ValueError('GLM-5.3 MTP expects the rotated W8A8 checkpoint')
        self.pp_group = get_pp_group()
        self.tp_size = get_parallel().tp_size
        self.num_fused_shared_experts = 0
        self.fuse_qkv_a_proj = self.config.q_lora_rank is not None
        self.use_dsa = True
        self.model = Glm53NextNModel(self.config, self.quant_config)
        self.lm_head = ParallelLMHead(self.config.vocab_size, self.config.hidden_size,
            quant_config=self.quant_config,
            prefix=f'model.layers.{self.config.num_hidden_layers}.shared_head.head',
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head)
        self.logits_processor = LogitsProcessor(self.config)
        self.capture_aux_hidden_states = False
        get_attn_tp_context().init_context(self.config.q_lora_rank, True)
        from .mtp_compat import patch_eagle_verify
        patch_eagle_verify()

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch):
        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            hidden = self.model(input_ids, positions, forward_batch)
        return self.logits_processor(input_ids, hidden, self.lm_head, forward_batch)

    def load_weights(self, weights):
        special = set()
        def decoder_weights():
            for name, tensor in weights:
                canonical = canonical_weight_name(name)
                if canonical == 'rot.weight':
                    default_weight_loader(self.model.rot.weight, tensor)
                    special.add('rot')
                elif canonical == f'model.layers.{self.config.num_hidden_layers}.embed_tokens.weight':
                    loader = getattr(self.model.embed_tokens.weight, 'weight_loader', default_weight_loader)
                    loader(self.model.embed_tokens.weight, tensor)
                    special.add('embed')
                elif canonical == f'model.layers.{self.config.num_hidden_layers}.shared_head.head.weight':
                    loader = getattr(self.lm_head.weight, 'weight_loader', default_weight_loader)
                    loader(self.lm_head.weight, tensor)
                    special.add('head')
                else:
                    yield name, tensor
        super().load_weights(decoder_weights(), is_nextn=True)
        if special != {'rot', 'head', 'embed'}:
            raise ValueError(f'Incomplete MTP rotation/head weights: {special}')
        logger.info('GLM MTP rotation and checkpoint-specific output head loaded')

    def set_embed_and_head(self, embed, head):
        # EAGLE calls this generically, but this checkpoint explicitly stores
        # MTP embeddings and a separate output head. Retain both loaded tensors.
        pass

    @property
    def routed_experts_weights_of_layer(self):
        return {0: self.model.decoder.mlp.get_moe_weights()}


EntryClass = [Glm5NextForConditionalGenerationNextN]
