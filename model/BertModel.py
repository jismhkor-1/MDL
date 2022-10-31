import math

from torch.nn import BCEWithLogitsLoss
from transformers import BertModel
import torch.nn.functional as F

from utils import *


class BertFGSC(nn.Module):
    def __init__(self, tokenizer, args):
        super().__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.bert_path)
        self.hidden_size = self.bert.config.hidden_size
        self.self_attn_dim = args.z_s_dim
        self.attn_projection = nn.Linear(self.hidden_size, self.self_attn_dim)

        self.K = args.category_num
        self.latent_feature_size = args.z_k_dim
        self.z_posterior = nn.Linear(self.hidden_size, self.latent_feature_size*self.K)
        self.ffns = [nn.Linear(self.latent_feature_size, 3) for _ in range(self.K)]
        self.tokenizer = tokenizer

    def forward(self, input_tokens, input_masks, clf_target=None):
        embbed_feature = self.bert.embeddings(input_tokens)
        attn_feature = self.attn_projection(embbed_feature)
        z_s_c = self.self_attention(attn_feature, input_masks)[:, 0, :]

        bert_feature, _ = self.bert(input_tokens, input_masks)
        bert_feature = bert_feature[:, 0, :]  # or max pool or mean pool
        posterior_out_z = self.z_posterior(bert_feature)
        mu_post_z, logvar_post_z = torch.chunk(posterior_out_z, 2, 1)
        sample_z_k_c = sample_from_gaussian(mu_post_z, logvar_post_z)

        output = []
        for i in range(self.K):
            predict = self.ffns[i](sample_z_k_c[:, i*self.latent_feature_size:(i+1)*self.latent_feature_size])
            output.append(predict)
        output = torch.cat(output, dim=1)
        if clf_target:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(output, clf_target)
            return FGSCOutputs(
                loss=loss,
                logits=output,
                z_s_c=z_s_c,
                z_k_c=sample_z_k_c,
                z_k_c_para=(mu_post_z, logvar_post_z)
            )

        return FGSCOutputs(
            logits=output,
            z_s_c=z_s_c,
            z_k_c=sample_z_k_c,
            z_k_c_para=(mu_post_z, logvar_post_z)
        )

    def self_attention(self, x, ipt_mask):
        d = x.shape[-1]
        attn_weights = torch.bmm(x, x.transpose(1, 2))
        v = F.tanh(attn_weights / math.sqrt(d))
        attn_mask = ipt_mask.eq(0)
        v.masked_fill(attn_mask, float("-inf"))
        attn_score = nn.Softmax(dim=-1)(v)
        output = torch.bmm(attn_score, x)
        return output
