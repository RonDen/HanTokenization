#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author:   Xukun Luo
# Date:     2021.04.09

import re
import argparse
import numpy as np

def get_pretraining_corpus(source_path, target_path, punc=False):
    """ 
        Generate the pretraining corpus based on the training text.

        Args:
        
            source_path:        The source path of training text.
            target_path:        The target path.
            punc:               If with punc.

        Returns:
    """
    fr = open(source_path, "r", encoding="utf-8")
    fw = open(target_path, "w", encoding="utf-8")

    #sub_str = r'，|。|、|“|”|（|）|：|；|—|《|》|．|『|』|…|！|-|’|／|∶|‘|－|●|ｒ|○|ａ|ｏ|▲|Ⅱ|⑵|〈|〉|①|②|③|°|④|⑤|⑥|⑦|＋|＝|＞|［|］|～|\?|]'
    # 去掉可以和其他字连成词的符号
    sub_str = r'，|。|、|“|”|（|）|：|；|《|》|『|』|！|’|∶|‘|●|ｒ|ａ|▲|Ⅱ|⑵|〈|〉|①|②|③|④|⑤|⑥|⑦|＋|＝|＞|［|］|～|\?|？|]'

    for index, _line in enumerate(fr):
        if punc is False:
            line = re.sub(sub_str, " ", _line)
        else:
            line = _line
        line = list(line.strip().replace(" ", ""))
        line = " ".join(line) + "\n"
        fw.write(line)

    fr.close()
    fw.close()

def add_reserved_vocab(embed_path, vocab_path, reserved_path, embed_size):
    """ 
        Add the reserved vocab words to embedding file and vaocab file.

        Args:

            embed_path:         The embedding path.
            vocab_path:         The vocabulary path.
            reserved_path:      The reserved path.
            embed_size:         The embedding vectors' size.

        Returns:
    """
    # Get reserved words.
    print("Getting total reserved words...")
    reversed_words = []
    with open(reserved_path, "r", encoding="utf-8") as fr:
        for index, line in enumerate(fr):
            line = line.strip()
            if line != "": reversed_words.append(line)
    
    # Add reserved vocab to the vacab file.
    print("Building new vocab file...")
    reversed_words_str = "\n".join(reversed_words) + "\n"
    with open(vocab_path, "r", encoding="utf-8") as fv:
        reversed_words_str += fv.read()
    with open(vocab_path, "w", encoding="utf-8") as fv:
        fv.write(reversed_words_str)

    # Add reserved vecters to the embed file.
    idx_embed = []
    idx_embed.extend([[0.0] * embed_size for _ in range(len(reversed_words))])
    print("Getting total Glove vectors...")
    with open(embed_path, "r", encoding="utf-8") as fe:
        for index, line in enumerate(fe):
            line = line.strip().split()
            word, vector = line[0], line[1:]
            vector = [float(item) for item in vector]
            assert len(vector) == embed_size
            idx_embed.append(vector)
    # For <unk> -> [UNK]
    idx_embed[1] = idx_embed[-1]
    idx_embed = idx_embed[0:-1]
    print("Building new embed file...")
    np.savetxt(embed_path, idx_embed)
    print("Add successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--type", 
        choices=["corpus", "reserved", "corpus_with_punc", "reserved_with_punc"],
        default="corpus", 
        help="What tool do you want to use.")

    args = parser.parse_args()

    if args.type == "corpus":
        source_path = "../../datasets/training.txt"
        target_path = "./corpus.txt"
        get_pretraining_corpus(source_path, target_path)
    elif args.type == "reserved":
        embed_path = "./glove_embed.txt"
        vocab_path = "./vocab.txt"
        reserved_path = "../reserved_vocab.txt"
        embed_size = 300
        add_reserved_vocab(embed_path, vocab_path, reserved_path, embed_size)
    elif args.type == "corpus_with_punc":
        source_path = "../../datasets/training.txt"
        target_path = "./corpus_with_punc.txt"
        get_pretraining_corpus(source_path, target_path, True)
    elif args.type == "reserved_with_punc":
        embed_path = "./glove_embed_with_punc.txt"
        vocab_path = "./vocab_with_punc.txt"
        reserved_path = "../reserved_vocab.txt"
        embed_size = 300
        add_reserved_vocab(embed_path, vocab_path, reserved_path, embed_size)