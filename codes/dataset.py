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
    def __init__(self, state=TRAIN, index=0, k_fold=K, vocab_path="../pretrained_models/Glove/vocab.txt"):
        """ 
            The Sighan Dataset.

            Args:

                train:          If the training set, the validation set or the test set. 
                                0->TRAIN, 1->VALID, 2->TEST.
                index:          The i-th fold in the dataset.
                k_fold:         Apply k-fold methodology.
                vocab_path:     The vocabulary path.

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
        if not os.path.exists(self.preprocessed_data_path):
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
                            print("Warning: the word length can not less than 1.")
                            continue
                    assert len(sentence) == len(label)
                    if len(sentence) > self.max_length:
                        self.max_length = len(sentence)
                    sentence_id = [self.vocab.get(t) for t in sentence]
                    if len(sentence) == 0: continue
                    data_str = json.dumps({"sentence": sentence, "sentence_id": sentence_id, "label": label}, ensure_ascii=False)
                    #print(data_str)
                    fw.write(data_str + "\n")
            
            fw.close()

            print("The max sentence length: {}.".format(self.max_length))
        
        # Get the data lines.
        fr_pre = open(self.preprocessed_data_path, "r", encoding="utf-8")
        data_lines = fr_pre.readlines()
        data_len = len(data_lines)

        if state == TRAIN:
            self.data_lines = data_lines[:int((index % k_fold) * data_len / k_fold)] + \
                                data_lines[int((index % k_fold + 1) * data_len / k_fold):]
        elif state == VALID:
            self.data_lines = data_lines[int((index % k_fold) * data_len / k_fold): \
                                int((index % k_fold + 1) * data_len / k_fold)]
        else:
            self.data_lines = data_lines

        fr_pre.close()

        # Get the sentences and labels.
        self.sentence, self.sentence_id, self.label = [], [], []
        for line in self.data_lines:
            line = line.strip()
            if line == "": continue
            line = json.loads(line, encoding="utf-8")
            self.sentence.append(line["sentence"])
            self.sentence_id.append(line["sentence_id"])
            self.label.append(line["label"])
        self.length = len(self.sentence)
        
        return
    
    def __getitem__(self, index):
        return self.sentence[index], self.sentence_id[index], self.label[index]
    
    def __len__(self):
        return self.length

def collate_fn(batch):
    """ 
        Collate_fn function. Arrange the batch in reverse order of length.

        Args:

            batch:              The batch data.
        
        Returns:
            ([sentence], [padded_sentence_id], [sentence_length], [padded_label])
    """
    batch.sort(key=lambda b: len(b[0]), reverse=True)
    data_length = [len(b[0]) for b in batch]
    sent_seq = [b[0] for b in batch]
    sent_id_seq = [torch.FloatTensor(b[1]) for b in batch]
    label = [torch.FloatTensor(b[2]) for b in batch]
    padded_sent_id_seq = pad_sequence(sent_id_seq, batch_first=True, padding_value=PAD_ID).long()
    padded_label = pad_sequence(label, batch_first=True, padding_value=label_dict["O"]).long()
    return sent_seq, padded_sent_id_seq, data_length, padded_label

if __name__ == "__main__":
    sighan_dataset = SighanDataset(TEST, 0, 10)
    sighan_data_loader = DataLoader(sighan_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    for batch in sighan_data_loader:
        print(batch[0])
        print(batch[1])
        print(batch[2])
        print(batch[3])
        break