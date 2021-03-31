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
from codes.config import *

""" SIGHAN Dataset. """

class SighanDataset(Dataset):
    def __init__(self, state=TRAIN, index=0, k_fold=K):
        """ 
            The Sighan Dataset.

            Args:

                train:          If the training set, the validation set or the test set. 
                                0->TRAIN, 1->VALID, 2->TEST.
                index:          The i-th fold in the dataset.
                k_fold:         Apply k-fold methodology.

            Returns:
        """
        self.origin_train_path = origin_train_path
        self.origin_test_path = origin_test_path
        self.preprocessed_train_path = preprocessed_train_path
        self.preprocessed_test_path = preprocessed_test_path
        
        self.origin_data_path = self.origin_test_path if state == TEST else self.origin_train_path
        self.preprocessed_data_path = self.preprocessed_test_path if state == TEST else self.preprocessed_train_path

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
                    data_str = json.dumps({"sentence": sentence, "label": label}, ensure_ascii=False)
                    #print(data_str)
                    fw.write(data_str + "\n")
            
            fw.close()
        
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
        self.sentence, self.label = [], []
        for line in self.data_lines:
            line = line.strip()
            if line == "": continue
            line = json.loads(line, encoding="utf-8")
            self.sentence.append(line["sentence"])
            self.label.append(line["label"])
        self.length = len(self.sentence)
        
        return
    
    def __getitem__(self, index):
        return self.sentence[index], self.label[index]
    
    def __len__(self):
        return self.length
        

if __name__ == "__main__":
    dataset = SighanDataset(VALID)
    sentence, label = dataset[5]
    print("length: {},\nsentence: {},\nlabel: {}.".format(len(dataset), sentence, label))