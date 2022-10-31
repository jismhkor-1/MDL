import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional, Tuple

from transformers.file_utils import ModelOutput


class LossOutputs(ModelOutput):
    clf_loss: Optional[torch.FloatTensor] = None
    gen_loss: Optional[torch.FloatTensor] = None
    dual_loss: Optional[torch.FloatTensor] = None
    rec_sentiment_loss: Optional[torch.FloatTensor] = None
    rec_text_loss: Optional[torch.FloatTensor] = None
    fgsc_cross_loss: Optional[torch.FloatTensor] = None
    fgsg_cross_loss: Optional[torch.FloatTensor] = None
    fgsc_disentanglement_loss: Optional[torch.FloatTensor] = None
    fgsg_disentanglement_loss: Optional[torch.FloatTensor] = None


class FGSCOutputs(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    z_s_c: torch.FloatTensor = None
    z_k_c: torch.FloatTensor = None
    z_k_c_para: Optional[(torch.FloatTensor, torch.FloatTensor)] = None


class FGSGOutputs(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    z_k_g: torch.FloatTensor = None
    z_s_g: torch.FloatTensor = None
    z_s_g_para: Optional[(torch.FloatTensor, torch.FloatTensor)] = None


def sample_from_gaussian(mu, logvar, seed=None):
    if seed is None:
        epsilon = logvar.new_empty(logvar.size()).normal_()  # train
    else:  # during generation set different random seed
        if not logvar.is_cuda:  # if it is not on gpu
            epsilon = logvar.new_empty(logvar.size()).normal_(generator=torch.manual_seed(seed))
        else:  # if tensor is on gpu
            epsilon = logvar.new_empty(logvar.size()).normal_(generator=torch.cuda.manual_seed_all(seed))
    std = torch.exp(0.5 * logvar)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    z = mu + std * epsilon
    return z

def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * torch.sum(1 + (recog_logvar - prior_logvar) -
                           torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar)) -
                           torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)), 1)
    return kld
