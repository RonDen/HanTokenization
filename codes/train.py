#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Xukun Luo
# Date:     2021.04.11

import sys
sys.path.append("../")

import os
import json
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from codes.vocab import Vocab
from codes.config import *
from codes.models import ModelDict
from codes.dataset import SighanDataset, collate_fn

# Seed.
def set_seed(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# Model saver
def save_model_with_optim(model, optimizer, model_path):
    if hasattr(model, "module"):
        state_dict = {"net": model.module.state_dict(), "optimizer": optimizer.state_dict()}
    else:
        state_dict = {"net": model.state_dict(), "optimizer": optimizer.state_dict()}
    
    torch.save(state_dict, model_path)

# k-fold file name.
def get_k_file_path(file_path, k_idx):
    file_name, extension = os.path.splitext(file_path)
    return file_name + str(k_idx) + extension

# Parameters loader.
def load_parameters():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_embedding_path", default="../pretrained_models/Glove/glove_embed.txt", type=str,
                        help="Path of the pretrained embedding file.")
    parser.add_argument("--last_model_path", default="../models/bilstm/last/model.bin", type=str,
                        help="Path of the output last output model.")
    parser.add_argument("--best_model_path", default="../models/bilstm/best/model.bin", type=str,
                        help="Path of the output best output model.")
    parser.add_argument("--middle_model_path", default=None, type=str,
                        help="Path of the middle model input path.")
    parser.add_argument("--result_path", default="../results/bilstm/test_result.txt", type=str,
                        help="Path of the middle model input path.")
    parser.add_argument("--vocab_path", default="../pretrained_models/Glove/vocab.txt", type=str,
                        help="Path of the vocabulary file.")

    # Model options.
    parser.add_argument("--model_type", 
        choices=["bilstm", "bilstmcrf", "transformer"],
        default="bilstm",
        help="What kind of model do you want to use.")
    parser.add_argument("--batch_size", type=int, default=batch_size,
                        help="Batch_size.")
    parser.add_argument("--hidden_size", type=int, default=hidden_size,
                        help="Hidden_size for transformers.")
    parser.add_argument("--feedforward_size", type=int, default=feedforward_size,
                        help="Feedforward_size for transformers.")
    parser.add_argument("--heads_num", type=int, default=heads_num,
                        help="Heads_num for transformers.")
    parser.add_argument("--transformer_layers", type=int, default=transformer_layers,
                        help="Transformer layers.")
    parser.add_argument("--seq_length", default=seq_length, type=int,
                        help="Sequence length.")
    
    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=learning_rate,
                        help="Learning rate.")
    
    # Training options.
    parser.add_argument("--dropout", type=float, default=dropout,
                        help="Dropout.")
    parser.add_argument("--lstm_layers", type=int, default=lstm_layers,
                        help="LSTM layers.")
    parser.add_argument("--lstm_hidden", type=int, default=lstm_hidden,
                        help="LSTM hidden size.")
    parser.add_argument("--epochs_num", type=int, default=epochs_num,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=report_steps,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=seed,
                        help="Random seed.")
    parser.add_argument("--K", type=int, default=K,
                        help="K fold.")
    parser.add_argument("--merge", type=int, default=0,
                        help="If merge. 0->False, 1->True")
    parser.add_argument("--separate", type=int, default=0,
                        help="If separate. 0->False, 1->True")
    
    args = parser.parse_args()

    args.lstm_dropout = args.dropout
    args.merge = True if args.merge == 1 else False
    args.separate = True if args.separate == 1 else False

    # Labels list.
    args.label_dict, args.label_list, args.label_number = label_dict, label_list, label_number

    # Vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab_len = len(vocab)

    if torch.cuda.is_available(): args.use_cuda = True
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args

# Model builder.
def build_model(args):
    # Get Glove Embedding.
    print("Loading the Glove Embedding...")
    embed_weight = torch.from_numpy(np.loadtxt(args.pretrained_embedding_path))
    args.embed_size = embed_weight.size(-1)
    print("Embedding size: ", args.embed_size)

    # Build sequence labeling model.
    model = ModelDict[args.model_type](args)
    #model.embedding.weight.data.copy_(embed_weight)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
        model = model.module

    # Load middle model.
    args.state_dict = {}
    if args.middle_model_path is not None:
        print("There is a middle model, let's use it!")
        args.state_dict = torch.load(args.middle_model_path)
        model.load_state_dict(args.state_dict["net"], strict=False)
    
    model = model.to(args.device)

    return model

# Evaluation function.
def evaluate(model, args, is_test, k_idx=None):
    update_flag = True if (k_idx is None or k_idx == 0) else False
    if is_test:
        sighan_dataset = SighanDataset(TEST, update=update_flag, merge=args.merge, separate=args.separate)
        # When evaluating the test set, the batch must be 1.
        sighan_data_loader = DataLoader(sighan_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        if k_idx is not None:
            result_file = get_k_file_path(args.result_path, k_idx)
        else:
            result_file = args.result_path
        fw = open(result_file, "w", encoding="utf-8")
    else:
        assert k_idx is not None
        sighan_dataset = SighanDataset(VALID, k_idx, args.K, update=update_flag, merge=args.merge, separate=args.separate)
        sighan_data_loader = DataLoader(sighan_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    correct, gold_number, pred_number = 0, 0, 0

    model.eval()

    for i, batch in enumerate(sighan_data_loader):

        sents, sent_ids, sent_lens, sent_labels, line_ids = batch
        sent_ids = sent_ids.to(args.device)

        feats = model(sent_ids, sent_lens)
        if hasattr(model, "get_best_path"):
            #print("Using the get_best_path func in the model...")
            best_path = model.get_best_path(feats, sent_lens)
        else:
            #print("Using the default get_best_path func...")
            best_path = torch.argmax(feats, dim = -1).view(-1, sent_ids.size(1))

        for j in range(0, len(sent_ids)):
            text, gold_tags = sents[j], [str(args.label_list[int(p)]) for p in sent_labels[j]]
            pred_tags = [str(args.label_list[p]) for p in best_path[j]]
            text_len = len(text)

            """ if i == 0 and j == 0:
                print("text: ", text) """

            """ Save the results. """
            if is_test:
                #if i == 0 and j == 0: print("Write the result.")
                if args.separate is False:
                    result_str = text[0]
                else:
                    result_str = str(int(line_ids[j])) + "[SEP]" + text[0]
                for k in range(1, text_len):
                    if pred_tags[k] == "S" or pred_tags[k] == "B":
                        result_str += ("  " + text[k])
                    else:
                        result_str += text[k]
                fw.write(result_str + "\n")

            """ Evaluate. """
            for k in range(text_len):
                # Gold.
                if gold_tags[k] == "S" or gold_tags[k] == "B":
                    gold_number += 1
                # Predict.
                if pred_tags[k] == "S" or pred_tags[k] == "B":
                    pred_number += 1
            
            pred_pos, gold_pos = [], []
            start, end = 0, 0
            # Correct.
            for k in range(text_len):
                if gold_tags[k] == "S":
                    start = k
                    end = k + 1
                elif gold_tags[k] == "B":
                    start = k
                    end = k + 1
                    while end < text_len:
                        if gold_tags[end] == "I": end += 1
                        elif gold_tags[end] == "E":
                            end += 1
                            break
                        else: break
                else:
                    continue
                gold_pos.append((start, end))
            # Predict
            for k in range(text_len):
                if pred_tags[k] == "S":
                    start = k
                    end = k + 1
                elif pred_tags[k] == "B":
                    start = k
                    end = k + 1
                    while end < text_len:
                        if pred_tags[end] == "I": end += 1
                        elif pred_tags[end] == "E":
                            end += 1
                            break
                        else: break
                else:
                    continue
                pred_pos.append((start, end))
            
            for pair in pred_pos:
                if pair in gold_pos: correct += 1
                """ if pair not in gold_tags: continue
                for k in range(pair[0], pair[1]):
                    if gold_tags[k] != pred_tags[k]: 
                        break
                else: 
                    correct += 1 """

    if is_test:
        fw.close()
        if args.separate is True:
            final_result = []
            with open(result_file, "r", encoding="utf-8") as f_result:
                for _, line in enumerate(f_result):
                    line = line.strip().split("[SEP]")
                    assert len(line) == 2
                    if int(line[0]) > len(final_result):
                        final_result.append(line[1])
                    else:
                        final_result[int(line[0]) - 1] += ("  " + line[1])
            with open(result_file, "w", encoding="utf-8") as f_result:
                for result_item in final_result:
                    f_result.write(result_item + "\n")
            
    p = correct / pred_number if pred_number != 0 else 0
    r = correct / gold_number
    f1 = 2*p*r/(p+r) if p != 0 else 0
    print("total_right, total_predict, predict_right: {}, {}, {}".format(gold_number, pred_number, correct))
    print("precision, recall, and f1: {:.3f}, {:.3f}, {:.3f}".format(p, r, f1))

    return f1

# Training function.
# If args.K > 1, we apply K-fold validation.
# If args.K == 1, we use all of training data to train the best model.
def train_kfold(args):
    # Training phase.
    print("Start training.")
    
    for k_idx in range(args.K):
        total_loss, f1, best_f1 = 0., 0., 0.

        model = build_model(args)

        # Evaluate the middle model.
        if args.middle_model_path is not None and k_idx == 0:
            print("Start evaluate middle model.")
            best_f1 = evaluate(model, args, True)

        # Criterion.
        default_criterion_flag = True
        if hasattr(model, "criterion"):
            print("Using the criterion func in the model...")
            criterion = model.criterion
            default_criterion_flag = False
        else:
            print("Using the default criterion func...")
            weight = [50.0] * args.label_number
            weight[label_dict["O"]] = 1.0
            weight = torch.tensor(weight).to(args.device)
            criterion = CrossEntropyLoss(weight, reduction="mean")

        # Optimizer.
        optimizer = Adam(model.parameters())
        if args.middle_model_path is not None:
            print("Loading optimizer...")
            optimizer.load_state_dict(args.state_dict["optimizer"])

        print("--------------- The {}-th fold as the validation set... ---------------".format(k_idx+1))

        # Get the training data.
        update_flag = True if k_idx == 0 else False
        sighan_dataset = SighanDataset(TRAIN, k_idx, args.K, update=update_flag, merge=args.merge, separate=args.separate)
        sighan_data_loader = DataLoader(sighan_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

        if k_idx == 0:
            print("Data length: ", len(sighan_dataset))

        for epoch in range(1, args.epochs_num + 1):
            model.train()
            for i, batch in enumerate(sighan_data_loader):
                model.zero_grad()

                sents, sent_ids, sent_lens, sent_labels, line_ids = batch
                sent_ids = sent_ids.to(args.device)
                sent_labels = sent_labels.to(args.device)

                feats = model(sent_ids, sent_lens)

                if default_criterion_flag:
                    loss = criterion(feats.contiguous().view(-1, args.label_number), sent_labels.view(-1))
                else:
                    loss = criterion(feats, sent_lens, sent_labels)

                if (i + 1) % args.report_steps == 0:
                    print("Epoch id: {}, Training steps: {}, Loss: {:.6f}".format(epoch, i+1, loss))

                loss.backward()
                optimizer.step()

            """ if epoch == 1:
                save_model_with_optim(model, optimizer, get_k_file_path(args.best_model_path, k_idx)) """

            if args.K > 1:
                f1 = evaluate(model, args, False, k_idx)
                if f1 >= best_f1:
                    best_f1 = f1
                    save_model_with_optim(model, optimizer, get_k_file_path(args.best_model_path, k_idx))

        # Save the last optimizer and model.
        save_model_with_optim(model, optimizer, get_k_file_path(args.last_model_path, k_idx))

        # Evaluation phase.
        print("Start evaluation.")

        if args.K > 1:
            model.load_state_dict(torch.load(get_k_file_path(args.best_model_path, k_idx)), strict=False)
        else:
            model.load_state_dict(torch.load(get_k_file_path(args.last_model_path, k_idx)), strict=False)

        evaluate(model, args, True, k_idx)

def main():
    args = load_parameters()
    set_seed(args.seed)
    train_kfold(args)

if __name__ == "__main__":
    main()