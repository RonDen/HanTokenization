#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Xukun Luo
# Date:     2021.03.31

import sys
sys.path.append("../")

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from codes.config import *
from codes.vocab import Vocab

""" SIGHAN Dataset. """

class SighanDataset(Dataset):
    def __init__(self, state=TRAIN, index=0, k_fold=K, vocab_path="../pretrained_models/Glove/vocab.txt", update=False, merge=False, separate=False):
        """ 
            The Sighan Dataset.

            Args:

                train:          If the training set, the validation set or the test set. 
                                0->TRAIN, 1->VALID, 2->TEST.
                index:          The i-th fold in the dataset.
                k_fold:         Apply k-fold methodology.
                vocab_path:     The vocabulary path.
                update:         To update the json file.
                separate:       Separate the text based on punc.

            Returns:
        """
        self.origin_train_path = origin_train_path
        self.origin_test_path = origin_test_path
        self.preprocessed_train_path = preprocessed_train_path
        self.preprocessed_test_path = preprocessed_test_path
        
        self.origin_data_path = self.origin_test_path if state == TEST else self.origin_train_path
        self.preprocessed_data_path = self.preprocessed_test_path if state == TEST else self.preprocessed_train_path

        self.max_length = 0
        self.vocab = Vocab()
        self.vocab.load(vocab_path, True)
        self.vocab_len = len(self.vocab)

        assert index < k_fold and index >= 0
        assert state == TRAIN or state == VALID or state == TEST

        # Generate the preprocessed data.
        if not os.path.exists(self.preprocessed_data_path) or update is True:
            #print("Building dataset...")
            fw = open(self.preprocessed_data_path, "w", encoding="utf-8")

            with open(self.origin_data_path, "r", encoding="utf-8") as fr:
                for idx, line in enumerate(fr):
                    sentence, label = "", []
                    line = line.strip().split("  ")
                    for word in line:
                        word_length = len(word)
                        sentence += word
                        if word_length == 1: label.append(label_dict["S"])
                        elif word_length > 1:
                            word_label = [label_dict["B"]] + [label_dict["I"]] * (word_length - 2) + [label_dict["E"]]
                            label.extend(word_label)
                        else:
                            #print("Warning: the word length can not less than 1.")
                            continue
                    assert len(sentence) == len(label)
                    if merge == True:
                        sentence, sentence_train, label = self.merge_special_tokens([w for w in sentence], label)
                        # print(sentence, sentence_train, label)
                        # if idx > 0:
                        #     exit()
                    else:
                        sentence = [w for w in sentence]
                        sentence_train = sentence
                    sentence_id = [self.vocab.get(t, punc_flag) for t in sentence_train]
                    if len(sentence) == 0: continue
                    if separate is True:
                        sentence_list, sentence_train_list, sentence_id_list, label_list = self.separate_text(sentence, sentence_train, sentence_id, label)
                        for sen_idx in range(len(sentence_list)):
                            data_str = json.dumps({"line_id": idx+1, "sentence": sentence_list[sen_idx], "sentence_train": sentence_train_list[sen_idx], "sentence_id": sentence_id_list[sen_idx], "label": label_list[sen_idx]}, ensure_ascii=False)
                            fw.write(data_str + "\n")
                            if len(sentence_list[sen_idx]) > self.max_length:
                                self.max_length = len(sentence_list[sen_idx])
                            """ if len(sentence_list[sen_idx]) == 959:
                                print("line_id: {}, sentence: {}".format(idx+1, sentence_list[sen_idx])) """
                    else:
                        data_str = json.dumps({"line_id": idx+1, "sentence": sentence, "sentence_train": sentence_train, "sentence_id": sentence_id, "label": label}, ensure_ascii=False)
                        #print(data_str)
                        fw.write(data_str + "\n")
                        if len(sentence) > self.max_length:
                            self.max_length = len(sentence)
            
            fw.close()

            #print("The max sentence length: {}.".format(self.max_length))
        
        # Get the data lines.
        fr_pre = open(self.preprocessed_data_path, "r", encoding="utf-8")
        data_lines = fr_pre.readlines()
        data_len = len(data_lines)

        if state == TRAIN:
            if k_fold > 1:
                self.data_lines = data_lines[:int((index % k_fold) * data_len / k_fold)] + \
                                    data_lines[int((index % k_fold + 1) * data_len / k_fold):]
            else:
                self.data_lines = data_lines
        elif state == VALID:
            if k_fold > 1:
                self.data_lines = data_lines[int((index % k_fold) * data_len / k_fold): \
                                    int((index % k_fold + 1) * data_len / k_fold)]
            else:
                self.data_lines = []
        else:
            self.data_lines = data_lines

        fr_pre.close()

        # Get the sentences and labels.
        self.sentence, self.sentence_id, self.label, self.line_id = [], [], [], []
        for line in self.data_lines:
            line = line.strip()
            if line == "": continue
            line = json.loads(line, encoding="utf-8")
            self.sentence.append(line["sentence"])
            self.sentence_id.append(line["sentence_id"])
            self.label.append(line["label"])
            self.line_id.append(line["line_id"])
        self.length = len(self.sentence)
        
        return
    
    def separate_text(self, sentence, sentence_train, sentence_id, label):
        separate_punc_list = [_ for _ in separate_punc]
        start, end = 0, 1
        sentence_list, sentence_train_list, sentence_id_list, label_list = [], [], [], []
        while end < len(sentence):
            if end == len(sentence) - 1:
                sentence_list.append(sentence[start:end+1])
                sentence_train_list.append(sentence_train[start:end+1])
                sentence_id_list.append(sentence_id[start:end+1])
                label_list.append(label[start:end+1])
                break
            if sentence[end] in separate_punc_list:
                sentence_list.append(sentence[start:end+1])
                sentence_train_list.append(sentence_train[start:end+1])
                sentence_id_list.append(sentence_id[start:end+1])
                label_list.append(label[start:end+1])
                start = end + 1
            end += 1
        return sentence_list, sentence_train_list, sentence_id_list, label_list
    
    def merge_special_tokens(self, word_list, label):
        """ 
            Merge the special tokens in word_list.

            Args:

                word_list:              The word list.
                label:                  The label list.
            
            Returns:

                word_list_new:          list with original tokens.
                word_list_mask:         list with mask tokens. ([NUM], [ALP])
                label_new:              new label list
        """
        half_numbers_list = [n for n in half_numbers]
        full_numbers_list = [n for n in full_numbers]
        chinese_numbers_list = [c for c in chinese_numbers]
        alphabet_list = [a for a in alphabet]
        serial_punc_list = [p for p in serial_punc]

        word_list_new, word_list_mask, label_new = [], [], []

        begin, word_list_len = 0, len(word_list)

        while begin < word_list_len:
            token = word_list[begin]
            if token not in half_numbers_list and token not in full_numbers_list and token not in alphabet_list and token not in serial_punc_list and token not in chinese_numbers_list:
                word_list_new.append(token)
                word_list_mask.append(token)
                label_new.append(label[begin])
                begin += 1
                continue
            if token in half_numbers_list:
                end = begin + 1
                while end < word_list_len:
                    if word_list[end] in half_numbers_list:
                        end += 1
                    else: break
                word_list_new.append("".join(word_list[begin:end]))
                word_list_mask.append("[NUM]")
                """ if self.get_new_label(begin, end-1, label[begin], label[end-1]) == label_dict["O"]:
                    print(word_list[begin:end]) """
                label_new.append(self.get_new_label(begin, end-1, label[begin], label[end-1]))
                begin = end
            elif token in full_numbers_list:
                end = begin + 1
                while end < word_list_len:
                    if word_list[end] in full_numbers_list:
                        end += 1
                    else: break
                word_list_new.append("".join(word_list[begin:end]))
                word_list_mask.append("[NUM]")
                """ if self.get_new_label(begin, end-1, label[begin], label[end-1]) == label_dict["O"]:
                    print(word_list[begin:end]) """
                label_new.append(self.get_new_label(begin, end-1, label[begin], label[end-1]))
                begin = end
            elif token in chinese_numbers_list:
                end = begin + 1
                while end < word_list_len:
                    if word_list[end] in chinese_numbers_list:
                        end += 1
                    else: break
                word_list_new.append("".join(word_list[begin:end]))
                word_list_mask.append("[NUM]")
                """ if self.get_new_label(begin, end-1, label[begin], label[end-1]) == label_dict["O"]:
                    print(word_list[begin:end]) """
                label_new.append(self.get_new_label(begin, end-1, label[begin], label[end-1]))
                begin = end
            elif token in alphabet_list:
                end = begin + 1
                while end < word_list_len:
                    if word_list[end] in alphabet_list:
                        end += 1
                    else: break
                word_list_new.append("".join(word_list[begin:end]))
                word_list_mask.append("[ALP]")
                """ if self.get_new_label(begin, end-1, label[begin], label[end-1]) == label_dict["O"]:
                    print(word_list[begin:end]) """
                label_new.append(self.get_new_label(begin, end-1, label[begin], label[end-1]))
                begin = end
            elif token in serial_punc_list:
                end = begin + 1
                while end < word_list_len:
                    if word_list[end] in serial_punc_list:
                        end += 1
                    else: break
                word_list_new.append("".join(word_list[begin:end]))
                word_list_mask.append("[PUNC]")
                """ if self.get_new_label(begin, end-1, label[begin], label[end-1]) == label_dict["O"]:
                    print(word_list[begin:end]) """
                label_new.append(self.get_new_label(begin, end-1, label[begin], label[end-1]))
                begin = end
        
        return word_list_new, word_list_mask, label_new
    
    def get_new_label(self, begin_idx, end_idx, label_begin, label_end):
        if begin_idx == end_idx: return label_begin

        if label_begin == label_dict["B"]:
            if label_end == label_dict["I"]: return label_dict["B"]
            elif label_end == label_dict["E"]: return label_dict["S"]
            else: 
                # print("Error when getting new label! label_begin: {}, label_end: {}".format(label_begin, label_end))
                return label_dict["O"]
        elif label_begin == label_dict["I"]:
            if label_end == label_dict["I"]: return label_dict["I"]
            elif label_end == label_dict["E"]: return label_dict["E"]
            else:
                # print("Error when getting new label! label_begin: {}, label_end: {}".format(label_begin, label_end))
                return label_dict["O"]
        else:
            # Single or End
            # print("Error when getting new label! label_begin: {}, label_end: {}".format(label_begin, label_end))
            return label_dict["O"]

    def __getitem__(self, index):
        return self.sentence[index], self.sentence_id[index], self.label[index], self.line_id[index]
    
    def __len__(self):
        return self.length

def collate_fn(batch):
    """ 
        Collate_fn function. Arrange the batch in reverse order of length.

        Args:

            batch:              The batch data.
        
        Returns:
            ([sentence], [padded_sentence_id], [sentence_length], [padded_label], line_idx)
    """
    batch.sort(key=lambda b: len(b[0]), reverse=True)
    data_length = [len(b[0]) for b in batch]
    sent_seq = [b[0] for b in batch]
    sent_id_seq = [torch.FloatTensor(b[1]) for b in batch]
    label = [torch.FloatTensor(b[2]) for b in batch]
    line_id = [b[3] for b in batch]
    padded_sent_id_seq = pad_sequence(sent_id_seq, batch_first=True, padding_value=PAD_ID).long()
    padded_label = pad_sequence(label, batch_first=True, padding_value=label_dict["O"]).long()
    return sent_seq, padded_sent_id_seq, data_length, padded_label, line_id

if __name__ == "__main__":
    sighan_dataset = SighanDataset(TRAIN, 0, 5, update=True, merge=True, separate=True)
    sighan_data_loader = DataLoader(sighan_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    for batch in sighan_data_loader:
        print(batch[0])
        print(batch[1])
        print(batch[2])
        print(batch[3])
        break