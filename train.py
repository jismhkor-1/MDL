import argparse

import numpy as np
import torch
from torch.optim import Adam
from transformers import BertTokenizer, BartTokenizer

from data_utils.DataLoader import *
from model.LanguageModel import Ngram
from model.MDL import MutualDisentangleLearning

# def pre_train():
#     model = BertForPreTraining.from_pretrained(args.bert_path)
#     tokenizer = BertTokenizer.from_pretrained(args.bert_path)
#     optim = Adam(params=model.parameters, lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)
#     data = np.concatenate([FGSCDataLoader(tokenizer, t, 'bert', args) for t in ['train', 'dev', 'test']])
#     data_batch = Batcher({'fgsc': data}, args, is_eval=True, batch_size=16, mode='bert')
#     for i in range(20):
#         for token, mask in data_batch:
#             loss = model(token, mask)


def clf_metric(target, predict):
    p = torch.sum(predict)
    g = torch.sum(target)
    g_minus_p = torch.sum((target - predict).eq(1).int())
    p_minus_g = torch.sum((predict - target).eq(1).int())
    x = p - p_minus_g
    assert x + g_minus_p == g

    precision = x / p if p > 0 else 0
    recall = x / g if g > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return precision, recall, f1


def main():
    train_fgsc_data = FGSCDataLoader(bert_tokenizer, 'train', 'bert', args)
    train_fgsg_data = FGSGDataloader(bart_tokenizer, 'train', 'bart', args)
    lm = Ngram(train_fgsc_data.get_text())
    train_dataset = {'fgsc': train_fgsc_data, 'fgsg': train_fgsg_data}

    dev_fgsc_data = FGSCDataLoader(bert_tokenizer, 'dev', 'bert', args)
    dev_fgsg_data = FGSGDataloader(bart_tokenizer, 'dev', 'bart', args)
    dev_dataset = {'fgsc': dev_fgsc_data, 'fgsg': dev_fgsg_data}

    model = MutualDisentangleLearning(bert_tokenizer, bart_tokenizer, args)

    # step 1
    def step1():
        # fgsc
        best_f1 = -1
        patience = 0
        optimizer11 = Adam(model.parameters(), lr=1e-4, weight_decay=5e-5)
        train_batch11 = Batcher(train_dataset, args, mode='fgsc')
        dev_batch11 = Batcher(dev_dataset, args, is_eval=True, mode='fgsc')
        print("step1, fgsc")
        for e in range(args.epochs):
            print("{:>2}/{}".format(e+1, args.epochs), end='')
            for text_bert_ind, text_bert_msk, cat_sen_target in tqdm(train_batch11):
                output = model.model_clf(
                    input_tokens=text_bert_ind,
                    input_masks=text_bert_msk,
                    clf_target=cat_sen_target
                )
                loss = output.loss
                optimizer11.zero_grad()
                loss.backward()
                optimizer11.step()
            model.eval()
            with torch.no_grad():
                tgts = []
                prds = []
                for text_bert_ind, text_bert_msk, cat_sen_target in tqdm(dev_batch11):
                    output = model.model_clf(input_tokens=text_bert_ind, input_masks=text_bert_msk)
                    logits = output.logits
                    predict = (logits > 0.5).int()
                    tgts.append(cat_sen_target)
                    prds.append(predict)
            tgts, prds = torch.cat(tgts), torch.cat(prds)
            p, r, f1 = clf_metric(tgts, prds)
            print('p: {:.4f}, r: {:.4f}, f1: {:.4f}'.format(p, r, f1))
            if f1 < best_f1:
                best_f1 = f1
                save_path = f"{args.model_path}/mdl.ckpt"
                torch.save(model, save_path)
                patience = 0
            else:
                patience += 1
            model.train()
            if patience > 1:
                break
        print("step1 fgsc finish")

        best_loss = 100000
        patience = 0
        optimizer12 = Adam(model.parameters(), lr=1e-4, weight_decay=5e-5)
        train_batch12 = Batcher(train_dataset, args, mode='fgsg')
        dev_batch12 = Batcher(train_dataset, args, is_eval=True, mode='fgsg')
        print("step1, fgsg")
        for e in range(args.epochs):
            print("{:>2}/{}".format(e+1, args.epochs), end='')
            for text_bart_ind, text_bart_msk, surr_bart_ind, surr_bart_msk in tqdm(train_batch12):
                output = model.model_gen(
                    src_input=surr_bart_ind,
                    src_mask=surr_bart_msk,
                    tgt_output=text_bart_ind
                )
                loss = output.loss
                optimizer12.zero_grad()
                loss.backward()
                optimizer12.step()
            model.eval()
            with torch.no_grad():
                loss_sum = 0
                for text_bart_ind, text_bart_msk, surr_bart_ind, surr_bart_msk in dev_batch12:
                    output = model.model_gen(
                        src_input=surr_bart_ind,
                        src_mask=surr_bart_msk,
                        tgt_output=text_bart_ind
                    )
                    loss_sum += output.loss
            if loss_sum < best_loss:
                best_loss = loss_sum
                save_path = f"{args.model_path}/mdl.ckpt"
                torch.save(model, save_path)
                patience = 0
            else:
                patience += 1
            model.train()
            if patience > 1:
                break
        print("step1 fgsg finish")

    # step 2
    def step2():
        train_batch2 = Batcher(train_dataset, args, language_model=lm)
        dev_batch2 = Batcher(dev_dataset, args, language_model=lm, is_eval=True)
        optimizer2 = Adam(model.parameters(), lr=1e-4, weight_decay=5e-5)
        patience = 0
        best_loss = 100000
        print("step2")
        for e in range(args.epochs):
            print("{:>2}/{}".format(e + 1, args.epochs), end='')
            for text_prob, text_bert_ind, text_bert_msk, cat_sen_target, \
                   text_bart_ind, text_bart_msk, surr_bart_ind, surr_bart_msk in tqdm(train_batch2):
                loss = model(
                    text_bert_token_id=text_bert_ind,
                    text_bert_token_mask=text_bert_msk,
                    text_bart_token_id=text_bart_ind,
                    text_bart_token_mask=text_bart_msk,
                    sur_bart_token_id=surr_bart_ind,
                    sur_bart_token_mask=surr_bart_msk,
                    cat_sen_clf_target=cat_sen_target,
                    text_prob=text_prob,
                    step=2)
                optimizer2.zero_grad()
                loss.backward()
                optimizer2.step()
            model.eval()
            with torch.no_grad():
                loss_sum = 0
                for text_prob, text_bert_ind, text_bert_msk, cat_sen_target, \
                   text_bart_ind, text_bart_msk, surr_bart_ind, surr_bart_msk in dev_batch2:
                    loss = model(
                        text_bert_token_id=text_bert_ind,
                        text_bert_token_mask=text_bert_msk,
                        text_bart_token_id=text_bart_ind,
                        text_bart_token_mask=text_bart_msk,
                        sur_bart_token_id=surr_bart_ind,
                        sur_bart_token_mask=surr_bart_msk,
                        cat_sen_clf_target=cat_sen_target,
                        text_prob=text_prob,
                        step=2)
                    loss_sum += loss
            if loss_sum < best_loss:
                best_loss = loss_sum
                save_path = f"{args.model_path}/mdl.ckpt"
                torch.save(model, save_path)
                patience = 0
            else:
                patience += 1
            model.train()
            if patience > 1:
                break
        print("step2 finish")

    # step 3
    def step3():
        train_batch3 = Batcher(train_dataset, args, language_model=lm)
        dev_batch3 = Batcher(dev_dataset, args, language_model=lm, is_eval=True)
        optimizer3 = Adam(model.parameters(), lr=1e-4, weight_decay=5e-5)
        patience = 0
        best_loss = 100000
        print("step3")
        for e in range(args.epochs*3):
            print("{:>2}/{}".format(e + 1, args.epochs), end='')
            for text_prob, text_bert_ind, text_bert_msk, cat_sen_target, \
                text_bart_ind, text_bart_msk, surr_bart_ind, surr_bart_msk in tqdm(train_batch3):
                output = model(
                    text_bert_token_id=text_bert_ind,
                    text_bert_token_mask=text_bert_msk,
                    text_bart_token_id=text_bart_ind,
                    text_bart_token_mask=text_bart_msk,
                    sur_bart_token_id=surr_bart_ind,
                    sur_bart_token_mask=surr_bart_msk,
                    cat_sen_clf_target=cat_sen_target,
                    text_prob=text_prob,
                    step=3)
                if e % 3 == 0:
                    loss = output.clf_loss + output.gen_loss + output.dual_loss
                elif e % 3 == 1:
                    loss = output.rec_text_loss + output.fgsc_disentanglement_loss + output.fgsc_cross_loss
                else:
                    loss = output.rec_sentiment_loss + output.fgsg_disentanglement_loss + output.fgsg_cross_loss
                optimizer3.zero_grad()
                loss.backward()
                optimizer3.step()
            if e % 3 < 2:
                continue
            model.eval()
            with torch.no_grad():
                loss_sum = 0
                for text_prob, text_bert_ind, text_bert_msk, cat_sen_target, \
                    text_bart_ind, text_bart_msk, surr_bart_ind, surr_bart_msk in dev_batch3:
                    loss = model(
                        text_bert_token_id=text_bert_ind,
                        text_bert_token_mask=text_bert_msk,
                        text_bart_token_id=text_bart_ind,
                        text_bart_token_mask=text_bart_msk,
                        sur_bart_token_id=surr_bart_ind,
                        sur_bart_token_mask=surr_bart_msk,
                        cat_sen_clf_target=cat_sen_target,
                        text_prob=text_prob,
                        step=2)
                    loss_sum += loss
            if loss_sum < best_loss:
                best_loss = loss_sum
                save_path = f"{args.model_path}/mdl.ckpt"
                torch.save(model, save_path)
                patience = 0
            else:
                patience += 1
            model.train()
            if patience > 1:
                break
        print("step3 finish")


def test():
    test_fgsc_data = FGSCDataLoader(bert_tokenizer, 'test', 'bert', args)
    test_fgsg_data = FGSGDataloader(bart_tokenizer, 'test', 'bart', args)
    test_data = {'fgsc': test_fgsc_data, 'fgsg': test_fgsg_data}
    test_batch = Batcher(test_data.dataset, args, is_eval=True)
    model_path = f"{args.model_path}/mdl.ckpt"
    model = torch.load(model_path, map_location=args.device)
    model.eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./MAMS')
    parser.add_argument('--bart_path', type=str, default='/home/huangj/bart/bart-base')
    parser.add_argument('--bert_path', type=str, default='/home/huangj/bert/bert-base')
    parser.add_argument('--pt_bert_path', type=str, default='/home/huangj/bert/bert-mams')

    parser.add_argument('--model_path', type=str, default='savemodel')
    parser.add_argument('--result_log', type=str, default='result')
    parser.add_argument('--info', type=bool, default=True)
    parser.add_argument('--max_ipt_len', type=int, default=90)
    parser.add_argument('--max_tgt_len', type=int, default=90)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)

    parser.add_argument('--category_num', '-K', type=int, default=8)
    parser.add_argument('--z_s_dim', '-s', type=int, default=128)
    parser.add_argument('--z_k_dim', '-k', type=int, default=32)

    args = parser.parse_args()

    args.bert_pad_id = 0
    args.bart_pad_id = 1

    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    bart_tokenizer = BartTokenizer.from_pretrained(args.bart_path)
