import re


# UNK = '斖'
UNK = '@'

rules = [
    r'[○一二三四五六七八九０１２３４５７９８]{4}年',
    r'[○一二三四五六七八九十０１２３４５７９８]+[月日]',
    r'[０１２３４５７９８]+[／．·]*[０１２３４５７９８]*[亿万％]',
    r'’９８',
    r'——',
    r'……'
]

def _gen_rules(rules):
    s = '|'.join([r'(%s)' %r for r in rules])
    return re.compile(s)

han_com = _gen_rules(rules=rules)

def preprocess(sentence):
    tmp = han_com.findall(sentence)
    recov = []
    for t in tmp:
        for sub in t:
            if sub and sub in sentence:
                # idx = sentence.find(sub)
                recov.append(sub)
                sentence = sentence.replace(sub, UNK)
    return sentence, recov


def recov(sen, rec):
    res, idx = [], 0
    for word in sen:
        if word == UNK and idx < len(rec):
            res.append(rec[idx])
            idx += 1
        else:
            res.append(word)
    return res
