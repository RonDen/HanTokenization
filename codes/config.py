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
K = 5
# other settings.
TRAIN, VALID, TEST = 0, 1, 2
label_list = ["B", "I", "E", "S", "O"]
label_dict = {"B": 0, "I": 1, "E": 2, "S": 3, "O": 4}
label_number = 5
PAD_ID = 0
UNK_ID = 1
PUNC_ID = 5

### For training. ###
# Model options
batch_size = 32
#seq_length = 1019   # test: 626
# Optimizer options.
learning_rate = 1e-4
# Training options.
dropout = 0.1
lstm_layers = 2
lstm_hidden = 300
epochs_num = 50
report_steps = 100
seed = 7