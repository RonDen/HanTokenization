# -*- encoding:utf-8 -*-

import sys
sys.path.append("../")

import os
import torch
from multiprocessing import Pool
from codes.config import UNK_ID, PUNC_ID


def count_line(corpus_path):
    count = 0
    with open(corpus_path, mode="r", encoding="utf-8") as f:
        for line in f:
            count += 1
    return count


class Vocab(object):
    """
    """
    def __init__(self, punc=None):
        self.w2i = {} 
        self.i2w = []
        if punc is None:
            #self.punc = ["，", "。", "、", "“", "”", "（", "）", "：", "；", "—", "《", "》", "．", "『", "』", "…", "！", "-", "’", "／", "∶", "‘", "－", "●", "ｒ", "○", "ａ", "ｏ", "▲", "Ⅱ", "⑵", "〈", "〉", "①", "②", "③", "°", "④", "⑤", "⑥", "⑦", "＋", "＝", "＞", "［", "］", "～", "?", "]"]
            self.punc = ["，", "。", "、", "“", "”", "（", "）", "：", "；", "《", "》", "『", "』", "！", "’", "∶", "‘", "●", "ｒ", "▲", "Ⅱ", "⑵", "〈", "〉", "①", "②", "③", "④", "⑤", "⑥", "⑦", "＋", "＝", "＞", "［", "］", "～", "?", "？", "]"]
        else:
            self.punc = punc
        
    def load(self, vocab_path, is_quiet=False):
        with open(vocab_path, mode="r", encoding="utf-8") as reader:
            for index, line in enumerate(reader):
                try:
                    w = line.strip().split()[0]
                    self.w2i[w] = index
                    self.i2w.append(w)
                except:
                    self.w2i["???"+str(index)] = index
                    self.i2w.append("???"+str(index))
                    if not is_quiet:
                        print("Vocabulary file line " + str(index+1) + " has bad format token")
            assert len(self.w2i) == len(self.i2w)
        if not is_quiet:
            print("Vocabulary Size: ", len(self))

    def save(self, save_path):
        print("Vocabulary Size: ", len(self))
        with open(save_path, mode="w", encoding="utf-8") as writer:
            for w in self.i2w:
                writer.write(w + "\n")
        print("Vocabulary saving done.")

    def get(self, w, punc=False):
        if punc is False:
            if w in self.punc: return PUNC_ID
        return self.w2i.get(w, UNK_ID)
        
    def __len__(self):
        return len(self.i2w)
                