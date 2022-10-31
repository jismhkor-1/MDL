from transformers.modeling_bart import \
    PretrainedBartModel, BartEncoder, BartDecoder, shift_tokens_right, make_padding_mask, invert_mask, \
    fill_with_neg_inf, Attention, _reorder_buffer
from transformers.configuration_bart import BartConfig
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput
from utils import *


def _make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


def _prepare_bart_decoder_inputs(
        config, input_ids, decoder_input_ids=None, decoder_padding_mask=None, causal_mask_dtype=torch.float32
):
    """
    Prepare masks that ignore padding tokens in the decoder and a causal mask for the decoder if none are provided.
    This mimics the default behavior in fairseq. To override it pass in masks. Note: this is not called during
    generation
    """
    pad_token_id = config.pad_token_id
    if decoder_input_ids is None:
        decoder_input_ids = shift_tokens_right(input_ids, pad_token_id)
    bsz, tgt_len = decoder_input_ids.size()
    if decoder_padding_mask is None:
        decoder_padding_mask = make_padding_mask(decoder_input_ids, pad_token_id)
    else:
        decoder_padding_mask = invert_mask(decoder_padding_mask)
    if decoder_padding_mask is not None and decoder_padding_mask.shape[1] > 1:
        # never mask leading token, even if it is pad
        decoder_padding_mask[:, 0] = decoder_padding_mask[:, 1]
    tmp = fill_with_neg_inf(torch.zeros(tgt_len, tgt_len))
    mask = torch.arange(tmp.size(-1))
    tmp.masked_fill_(mask < (mask + 1).view(tmp.size(-1), 1), 0)
    causal_mask = tmp.to(dtype=causal_mask_dtype, device=decoder_input_ids.device)
    return decoder_input_ids, decoder_padding_mask, causal_mask


class BartSeq2Seq(PretrainedBartModel):
    def __init__(self, config: BartConfig, args):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.config = config
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.register_buffer("final_logits_bias", torch.zeros((1, self.shared.num_embeddings)))

        self.latent_feature_size = args.z_s_dim
        self.attn_feature_size = args.z_k_dim
        self.K = args.category_num
        self.z_posterior = nn.Linear(config.d_model, self.latent_feature_size * 2)
        self.attn_proj = nn.Linear(config.d_model, self.K * self.attn_feature_size, bias=False)
        self.self_attn = Attention(embed_dim=self.K * self.attn_feature_size,
                                   num_heads=self.K,
                                   dropout=config.attention_dropout)
        self.reverse_proj = nn.Linear(self.K * self.attn_feature_size + self.latent_feature_size + config.d_model,
                                      config.d_model, bias=False)
        self.init_weights()

    def _resize_final_logits_bias(self, new_num_tokens: int, old_num_tokens: int) -> None:
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        old_num_tokens = self.shared.num_embeddings
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self.shared = new_embeddings
        self._resize_final_logits_bias(new_num_tokens, old_num_tokens)
        return new_embeddings

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs: Optional[Tuple] = None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        **kwargs,
    ):
        if labels is not None:
            use_cache = False
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        assert decoder_input_ids is not None

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        decoder_state, z_k_g, z_s_g, z_s_g_sample_para = \
            self.from_last_hidden_state_to_decoder_initial_state(encoder_outputs[0], attention_mask)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            decoder_input_ids,
            # encoder_outputs[0],
            decoder_state,
            attention_mask,
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            outputs = decoder_outputs + encoder_outputs
        else:
            outputs = Seq2SeqModelOutput(
                last_hidden_state=decoder_outputs.last_hidden_state,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )

        lm_logits = F.linear(outputs[0], self.shared.weight, bias=self.final_logits_bias)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return FGSGOutputs(
            loss=masked_lm_loss,
            logits=lm_logits.float(),
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            z_k_g=z_k_g,
            z_s_g=z_s_g,
            z_s_g_para=z_s_g_sample_para
        )

    def from_last_hidden_state_to_decoder_initial_state(self, last_hidden_state, attention_mask):
        posterior_out_z = self.z_posterior(last_hidden_state[:, 0, :])
        mu_post_z, logvar_post_z = torch.chunk(posterior_out_z, 2, 1)
        sample_z_s_g = sample_from_gaussian(mu_post_z, logvar_post_z)

        attn_input = self.attn_proj(last_hidden_state)
        attn_input = attn_input.transpose(0, 1)
        attn_mask = invert_mask(attention_mask)
        z_k_g, _ = self.self_attn(query=attn_input, key=attn_input, key_padding_mask=attn_mask)
        z_k_g = z_k_g.transpose(0, 1)[:, 0, :]

        cat_feature = torch.cat([z_k_g, sample_z_s_g], dim=-1)
        decoder_initial_state = self.reverse_proj(
            torch.cat(
                (cat_feature.unsqueeze(1).repeat(1, last_hidden_state.shape[1], 1), last_hidden_state),
                dim=1
            )
        )
        return decoder_initial_state, z_k_g, sample_z_s_g, (mu_post_z, logvar_post_z)

    # def from_last_hidden_state_to_decoder_initial_state(self, last_hidden_state, attention_mask):
    #     return last_hidden_state

    def prepare_inputs_for_generation(
            self, decoder_input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        decoder_initial_state, _, _, _ = \
            self.from_last_hidden_state_to_decoder_initial_state(encoder_outputs, attention_mask)
        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            # "encoder_outputs": encoder_outputs,
            "encoder_outputs": decoder_initial_state,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.shared)  # make it on the fly

    def get_encoder(self):
        return self.encoder

    def _reorder_cache(self, past, beam_idx):
        reordered_past = []
        for layer_past in past:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: _reorder_buffer(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
            }
            reordered_past.append(layer_past_new)
        return reordered_past

    # def adjust_logits_during_generation(self, logits, cur_len, max_length):
    #     if cur_len == 1 and self.config.force_bos_token_to_be_generated:
    #         self._force_token_ids_generation(logits, self.config.bos_token_id)
    #     elif cur_len == max_length - 1 and self.config.eos_token_id is not None:
    #         self._force_token_ids_generation(logits, self.config.eos_token_id)
    #     return logits
    #
    # def _force_token_ids_generation(self, scores, token_id) -> None:
    #     """force one of token_ids to be generated by setting prob of all other tokens to 0 (logprob=-float("inf"))"""
    #     scores[:, [x for x in range(self.config.vocab_size) if x != token_id]] = -float("inf")


class BartFGSG(nn.Module):
    def __init__(self, tokenizer, args):
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.max_decode_len = args.max_tgt_len
        self.model = BartSeq2Seq.from_pretrained(args.bart_path, args)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, src_input, src_mask, tgt_output):
        src_mask = src_mask.type(src_input.type())
        tgt_ipt = shift_tokens_right(tgt_output, pad_token_id=1)
        tgt_msk = tgt_ipt.ne(1)
        tgt_label = torch.where(tgt_output.eq(1),
                                -100 * torch.ones(tgt_output.shape, dtype=torch.long, device=tgt_output.device),
                                tgt_output)
        return self.model(input_ids=src_input, attention_mask=src_mask,
                          decoder_input_ids=tgt_ipt, decoder_attention_mask=tgt_msk, labels=tgt_label)

    def generate(self, src_input, src_mask, n_beams=1, n_return_sequences=1, start_token=2):
        result_list = []
        outputs = self.model.generate(input_ids=src_input, attention_mask=src_mask, max_length=self.max_decode_len,
                                      num_beams=n_beams, num_return_sequences=n_return_sequences)
        for predicted_ids in outputs:
            one_result = self.tokenizer.decode(predicted_ids, skip_special_tokens=True)
            result_list.append(one_result)
