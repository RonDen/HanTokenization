import numpy as np
import os
import matplotlib.pyplot as plt
from collections import Counter

import matplotlib
matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号

from nltk.probability import FreqDist

DATAROOT = '../datasets'
RESULTROOT = '../results'
VOCAB_FILE = os.path.join(DATAROOT, 'training_vocab.txt')
TRAIN_FILE = os.path.join(DATAROOT, 'training.txt')
TEST_FILE = os.path.join(DATAROOT, 'test.txt')

vocab = set()
train_set = []
test_set = []

with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    vocab = set(map(str.strip, f.readlines()))

with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
    train_set = list(map(str.strip, f.readlines()))

with open(TEST_FILE, 'r', encoding='utf-8') as f:
    test_set = list(map(str.strip, f.readlines()))


train_set_split = [line.split('  ') for line in train_set]

cnt = Counter()
for line in train_set_split:
    cnt.update(line)


frqdist = FreqDist(cnt)


def get_train_vocab_freq():
    result_file = os.path.join(RESULTROOT, 'vocab-freq.txt')
    with open(result_file, 'w+', encoding='utf-8') as f:
        f.writelines(["%s %d\n" % (word, time) for word, time in cnt.most_common()])
        


def get_top80_word_freq(num=80):
    plt.figure(figsize=(16, 8))
    ax = frqdist.plot(num)
    plt.tight_layout()
    plt.savefig("wordcount80.pdf")
    plt.show()

    # plt.figure(figsize=(16, 8))
    # ax = frqdist.plot(num, cumulative=True)
    # plt.savefig("前80个词的词频-累加.png")
    # plt.show()

get_top80_word_freq()
# get_train_vocab_freq()
