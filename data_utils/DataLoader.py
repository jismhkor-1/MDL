import os
import pickle
import json
import numpy as np
import torch
from tqdm import tqdm
from surrogate import surrogate_generation
import random


def text_process(sentence):
    return sentence.replace(', ', ' , ').replace('.', ' .')


class DataProcessor(object):
    def __init__(self, tokenizer, dataset, model_type, args):
        self.tokenizer = tokenizer
        file_name = f'{args.data_path}/{dataset}_{model_type}.pkl'
        if not args.data_rebuild and os.path.exists(file_name):
            self.dataset = pickle.load(open(file_name, 'rb'))
        else:
            js_name = f'{args.data_path}/{dataset}.json'
            self.dataset = self.load_data(json.load(open(js_name, encoding='utf-8')))
            pickle.dump(self.dataset, open(file_name, 'wb'))

        if args.info:
            print(f"load {file_name.split('.')[0]}")
            self.info()

    def info(self):
        raise NotImplementedError

    def load_data(self, datafile):
        raise NotImplementedError

    def get_text(self):
        record = []
        for d in self.dataset:
            record.append(d['text'])
        return record


class FGSGDataloader(DataProcessor):
    def __init__(self, tokenizer, dataset, model_type, args):
        super().__init__(tokenizer, dataset, model_type, args)
        self.args = args

    def info(self):
        pass

    def load_data(self, datafile):
        dataset = []
        for d in tqdm(datafile):
            text = text_process(d['text'].lower())
            text_token_ids = self.tokenizer.encode(text)
            data = ''.join([f"<{pair['category']}:{pair['polarity']}>" for pair in d['category_sentiment']])
            surrogate = surrogate_generation(d['category_sentiment'])
            surrogate_token_ids = self.tokenizer.encode(surrogate)
            dataset.append({"text": text, "data": data, "surrogate": surrogate,
                            "text_token_id": text_token_ids,
                            "surrogate_token_id": surrogate_token_ids})
        return np.array(dataset)


class FGSCDataLoader(DataProcessor):
    def __init__(self, tokenizer, dataset, model_type, args):
        super().__init__(tokenizer, dataset, model_type, args)
        self.args = None

    def info(self):
        pass

    def load_data(self, datafile):
        data_set = []
        for d in datafile:
            text = text_process(d['text'].lower())
            text_token_ids = self.tokenizer.encode(text)
            data = ''.join([f"<{pair['category']}:{pair['polarity']}>" for pair in d['category_sentiment']])
            cat_sen = np.zeros(15)
            for cs in d['category_sentiment']:
                cid = cs['category_id']
                sid = cs['polarity_id']
                cat_sen[cid * 3 + sid] = 1
            data_set.append({'text': text, 'data': data,
                             'text_token_id': text_token_ids,
                             'category-sentiment_target': cat_sen})
        return np.array(data_set)


class Batcher(object):
    def __init__(self, dataset, args, language_model=None, is_eval=False, batch_size=None, mode='both'):
        self.data = dataset
        self.data_size = len(dataset)
        self.device = args.device
        if batch_size:
            self.batch_size = batch_size
        else:
            self.batch_size = args.batch_size
        self.max_ipt_len = args.max_ipt_len
        self.max_tgt_len = args.max_tgt_len
        self.eval = is_eval
        self.args = args
        self.mode = mode
        self.lm = language_model
        self._reset()

    def _reset(self):
        ids = list(range(self.data_size))
        if not self.eval:
            random.shuffle(ids)
        self.idx_pool = [ids[i: i + self.batch_size] for i in range(0, len(self.data), self.batch_size)]
        self.pointer = 0

    def _pad(self, data_bert, data_bart):
        assert len(data_bart) == len(data_bert)
        text_bert_ind = np.ones((len(data_bert), self.max_ipt_len)) * self.args.bert_pad_id
        text_bert_msk = np.zeros((len(data_bert), self.max_ipt_len))
        text_bart_ind = np.ones((len(data_bart), self.max_tgt_len)) * self.args.bart_pad_id
        text_bart_msk = np.zeros((len(data_bart), self.max_tgt_len))
        surr_bart_ind = np.ones((len(data_bart), self.max_ipt_len)) * self.args.bart_pad_id
        surr_bart_msk = np.zeros((len(data_bart), self.max_ipt_len))
        cat_sen_target = []
        text_prob = []

        for i in range(len(data_bart)):
            da, de = data_bart[i], data_bert[i]
            assert da['text'] == de['text']
            if self.lm:
                text_prob.append(self.lm.score(da['text']))
            ipl_bert = min(len(de['text_token_id']), self.max_ipt_len)
            ipl_bart = min(len(da['surrogate_token_id']), self.max_ipt_len)
            tgl_bart = min(len(da['text_token_id']), self.max_tgt_len)
            text_bert_ind[i, :ipl_bert] = de['text_token_id'][:ipl_bert]
            text_bert_msk[i, :ipl_bert] = 1
            text_bart_ind[i, :tgl_bart] = da['text_token_id'][:tgl_bart]
            text_bart_msk[i, :tgl_bart] = 1
            surr_bart_ind[i, :ipl_bart] = da['surrogate_token_id'][:ipl_bart]
            surr_bart_msk[i, :ipl_bart] = 1
            cat_sen_target.append(da['category-sentiment_target'])

        if self.mode == 'fgsc':
            text_bert_ind = torch.tensor(text_bert_ind, dtype=torch.long, device=self.device)
            text_bert_msk = torch.tensor(text_bert_msk, device=self.device)
            cat_sen_target = torch.tensor(np.stack(cat_sen_target), dtype=torch.long, device=self.device)
            return text_bert_ind, text_bert_msk, cat_sen_target
        elif self.mode == 'fgsg':
            text_bart_ind = torch.tensor(text_bart_ind, dtype=torch.long, device=self.device)
            text_bart_msk = torch.tensor(text_bart_msk, device=self.device)
            surr_bart_ind = torch.tensor(surr_bart_ind, dtype=torch.long, device=self.device)
            surr_bart_msk = torch.tensor(surr_bart_msk, device=self.device)
            return text_bart_ind, text_bart_msk, surr_bart_ind, surr_bart_msk
        else:
            text_bert_ind = torch.tensor(text_bert_ind, dtype=torch.long, device=self.device)
            text_bert_msk = torch.tensor(text_bert_msk, device=self.device)
            cat_sen_target = torch.tensor(np.stack(cat_sen_target), dtype=torch.long, device=self.device)
            text_bart_ind = torch.tensor(text_bart_ind, dtype=torch.long, device=self.device)
            text_bart_msk = torch.tensor(text_bart_msk, device=self.device)
            surr_bart_ind = torch.tensor(surr_bart_ind, dtype=torch.long, device=self.device)
            surr_bart_msk = torch.tensor(surr_bart_msk, device=self.device)
            text_prob = torch.tensor(text_prob, device=self.device)
            return text_prob, text_bert_ind, text_bert_msk, cat_sen_target, \
                   text_bart_ind, text_bart_msk, surr_bart_ind, surr_bart_msk

    def _pad_bert(self, data_bert):
        text_bert_ind = np.ones((len(data_bert), self.max_ipt_len)) * self.args.bert_pad_id
        text_bert_msk = np.zeros((len(data_bert), self.max_ipt_len))
        for i in range(len(data_bert)):
            ipl_bert = min(len(data_bert['text_token_id']), self.max_ipt_len)
            text_bert_ind[i, :ipl_bert] = data_bert['text_token_id'][:ipl_bert]
            text_bert_msk[i, :ipl_bert] = 1
        text_bert_ind = torch.tensor(text_bert_ind, dtype=torch.long, device=self.device)
        text_bert_msk = torch.tensor(text_bert_msk, device=self.device)
        return text_bert_ind, text_bert_msk

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.idx_pool):
            self._reset()
            raise StopIteration()

        idx = self.idx_pool[self.pointer]
        if self.mode == 'bert':
            data = self.data['fgsc'].dataset[idx]
            self.pointer += 1
            return self._pad_bert(data)

        data_fgsc = self.data['fgsc'].dataset[idx]
        data_fgsg = self.data['fgsg'].dataset[idx]
        self.pointer += 1
        return self._pad(data_fgsc, data_fgsg)

    def __len__(self):
        return len(self.idx_pool)
