import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch import norm
from model import BartModel, BertModel, Decoder
from model.utils import LossOutputs


def label_logprob(label):
    label_num = label.shape[-1]
    p = 1/label_num
    count = torch.sum(label, dim=-1)
    logprob = count * torch.log(p) + (label_num-count) * torch.log(1-p)
    return logprob


class MutualDisentangleLearning(nn.Module):
    def __init__(self, bert_tokenizer, bart_tokenizer, args):
        super().__init__()
        self.args = args
        self.bert_vocab_size = len(bert_tokenizer)
        self.bart_vocab_size = len(bart_tokenizer)
        self.model_clf = BertModel.BertFGSC(bert_tokenizer, args)
        self.model_gen = BartModel.BartFGSG(bart_tokenizer, args)
        self.recon_clf = [nn.Linear(args.z_k_dim, 3) for _ in args.category_num]
        self.recon_gen = Decoder.LstmDecoder(vocab_size=self.bert_vocab_size,
                                             embedding_size=300,
                                             input_size=args.z_s_dim)

    def forward(
            self,
            text_bert_token_id,
            text_bert_token_mask,
            text_bart_token_id,
            text_bart_token_mask,
            sur_bart_token_id,
            sur_bart_token_mask,
            cat_sen_clf_target,
            text_prob,
            step):
        clf_output = self.model_clf(input_tokens=text_bert_token_id,
                                    input_masks=text_bert_token_mask,
                                    clf_target=cat_sen_clf_target)
        gen_output = self.model_gen(src_input=sur_bart_token_id,
                                   src_mask=sur_bart_token_mask,
                                   tgt_output=text_bart_token_id)

        clf_logits = clf_output.logits
        gen_logits = gen_output.logits
        clf_given_text = torch.sum(torch.log(clf_logits) * cat_sen_clf_target, dim=-1)

        loss_fct = CrossEntropyLoss(reduce=False)
        batch_size = text_bart_token_id.shape[0]
        tgt_label = torch.where(text_bart_token_id.eq(1),
                                -100 * torch.ones(text_bart_token_id.shape,
                                                  dtype=torch.long, device=self.args.device),
                                text_bart_token_id)
        text_given_surr = loss_fct(gen_logits.view(batch_size, -1, self.bart_vocab_size), tgt_label)

        label_logprob = self.label_prob(cat_sen_clf_target)
        text_logprob = torch.log(text_prob)
        dual_loss = torch.sum(torch.absolute(text_logprob + clf_given_text - label_logprob - text_given_surr))

        if step == 2:
            loss = clf_output.loss + gen_output.loss + dual_loss
            return loss

        elif step == 3:
            # reconstruction
            z_k_g = gen_output.z_k_g
            sentiment_rec = []
            for i in range(self.args.category_num):
                predict = self.recon_clf[i](z_k_g[:, i * self.args.z_k_dim:(i + 1) * self.args.z_k_dim])
                sentiment_rec.append(predict)
            sentiment_rec = torch.cat(sentiment_rec, dim=1)
            loss_sentiment_rec = torch.sum(torch.log(sentiment_rec) * cat_sen_clf_target)

            z_s_c = clf_output.z_s_c
            logits = self.recon_gen(input_feature=z_s_c, tgt_token_id=text_bert_token_id)
            loss_fct = CrossEntropyLoss(ignore_index=0)
            loss_text_rec = loss_fct(logits.view(-1, self.bert_vocab_size), text_bert_token_id.view(-1))

            # decomposing

            # cross
            z_s_g = gen_output.z_s_g
            z_k_c = clf_output.z_k_c
            loss_cross_ref_k = torch.exp(
                norm(z_k_c, p=None, dim=1)*norm(z_k_g, p=None, dim=1) /
                torch.diagonal(torch.bmm(z_k_c, z_k_g.transpose(0, 1)))
            )
            loss_cross_ref_s = torch.exp(
                norm(z_s_c, p=None, dim=1)*norm(z_s_g, p=None, dim=1) /
                torch.diagonal(torch.bmm(z_s_c, z_s_g.transpose(0, 1)))
            )
            return LossOutputs(clf_loss=clf_output.loss,
                               gen_loss=gen_output.loss,
                               dual_loss=dual_loss,
                               rec_sentiment_loss=loss_sentiment_rec,
                               rec_text_loss=loss_text_rec,
                               fgsc_cross_loss=loss_cross_ref_k,
                               fgsg_cross_loss=loss_cross_ref_s,
                               fgsc_disentanglement_loss=None,
                               fgsg_disentanglement_loss=None
                               )
