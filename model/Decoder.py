import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers.modeling_bart import shift_tokens_right


class LstmDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, input_size, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, pad_id)
        self.decoder_cell = nn.LSTMCell(input_size=embedding_size, hidden_size=embedding_size)
        self.cell_input_projection = nn.Linear(in_features=embedding_size+input_size,
                                               out_features=embedding_size)
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.pad_id = pad_id

    def forward(self, input_feature, tgt_token_id):
        tgt_ipt = shift_tokens_right(tgt_token_id, pad_token_id=self.pad_id)
        tgt_emb = self.embedding(tgt_ipt)

        h_ = input_feature
        h0 = torch.zeros(h_.shape, device=h_.device)
        c0 = torch.zeros(h_.shape, device=h_.device)
        output = []
        for y_t in torch.split(tgt_emb, 1, dim=1):
            h0 = self.cell_input_projection(torch.cat([h_, h0], dim=-1))
            y = y_t.squeeze(1)
            h_, c0 = self.decoder_cell(y, (h0, c0))
            output.append(h_)
        output = torch.stack(output, 1)
        lm_logits = F.linear(output, self.embedding.weight)
        return lm_logits
