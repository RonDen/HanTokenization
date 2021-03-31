#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Xukun Luo
# Date:     2021.03.31

""" Our settings. """

### For dataset. ###
# origin file path.
origin_train_path = "../datasets/training.txt"
origin_test_path = "../datasets/test.txt"
vocab_path = "../datasets/training_vocab.txt"
# preprocessed file path.
preprocessed_train_path = "../datasets/training_pre.json"
preprocessed_test_path = "../datasets/test_pre.json"
# k-fold.
K = 10
# other settings.
TRAIN, VALID, TEST = 0, 1, 2
label_list = ["B", "I", "E", "S"]
label_dict = {"B": 0, "I": 1, "E": 2, "S": 3}