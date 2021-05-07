import re
import os
from prob import trans_P, emit_P, start_P
from preprocess import preprocess, recov, UNK


DATAROOT = '/home/luod/class/nlp/HanTokenization/datasets'
RESULTROOT = '/home/luod/class/nlp/HanTokenization/results'
VOCAB_FILE = os.path.join(DATAROOT, 'training_vocab.txt')
VOCAB_FREQ = os.path.join(RESULTROOT, 'vocab-freq.txt')
TRAIN_FILE = os.path.join(DATAROOT, 'training.txt')
TEST_FILE = os.path.join(DATAROOT, 'test.txt')


MIN_FLOAT = -3.14e100


PrevStatus = {
    'B': 'ES',
    'M': 'MB',
    'S': 'SE',
    'E': 'BM'
}

Force_Split_Words = set([])
def add_force_split(word):
    global Force_Split_Words
    Force_Split_Words.add(word)


def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]  # tabular
    path = {}
    for y in states:  # init
        V[0][y] = start_p[y] + emit_p[y].get(obs[0], MIN_FLOAT)
        path[y] = [y]
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}
        for y in states:
            em_p = emit_p[y].get(obs[t], MIN_FLOAT)
            (prob, state) = max(
                [(V[t - 1][y0] + trans_p[y0].get(y, MIN_FLOAT) + em_p, y0) for y0 in PrevStatus[y]])
            V[t][y] = prob
            newpath[y] = path[state] + [y]
        path = newpath

    (prob, state) = max((V[len(obs) - 1][y], y) for y in 'ES')

    return (prob, path[state])


def hmm_cut(sentence):
    global emit_P
    prob, pos_list = viterbi(sentence, 'BMES', start_P, trans_P, emit_P)
    begin, nexti = 0, 0
    # print pos_list, sentence
    for i, char in enumerate(sentence):
        pos = pos_list[i]
        if pos == 'B':
            begin = i
        elif pos == 'E':
            yield sentence[begin:i + 1]
            nexti = i + 1
        elif pos == 'S':
            yield char
            nexti = i + 1
    if nexti < len(sentence):
        yield sentence[nexti:]

re_han = re.compile("([\u4E00-\u9FD5]+)")
re_skip = re.compile("([a-zA-Z0-9]+(?:\.\d+)?%?)")


def cut(sentence):
    if not sentence:
        yield None
    blocks = re_han.split(sentence)
    for blk in blocks:
        if re_han.match(blk):
            for word in hmm_cut(blk):
                if word not in Force_Split_Words:
                    yield word
                else:
                    for c in word:
                        yield c
        else:
            tmp = re_skip.split(blk)
            for x in tmp:
                if x:
                    yield x


with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
    train_set = list(map(str.strip, f.readlines()))

with open(TEST_FILE, 'r', encoding='utf-8') as f:
    test_set = list(map(str.strip, f.readlines()))

train_set_split = [line.split('  ') for line in train_set]
test_set_split = [line.split('  ') for line in test_set]


train_raw = [''.join(line) for line in train_set_split]
test_raw = [''.join(line) for line in test_set_split]


def eval(file_path, train=False):
    if not train:
        os.system('perl /home/luod/class/nlp/HanTokenization/scripts/score /home/luod/class/nlp/HanTokenization/datasets/training_vocab.txt /home/luod/class/nlp/HanTokenization/datasets/test.txt %s ' % file_path)
    else:
        os.system('perl /home/luod/class/nlp/HanTokenization/scripts/score /home/luod/class/nlp/HanTokenization/datasets/training_vocab.txt /home/luod/class/nlp/HanTokenization/datasets/training.txt %s' % file_path)


def pre_make_cut(cut_func, result_file):
    file_path = os.path.join(RESULTROOT, result_file)
    with open(os.path.join(RESULTROOT, result_file), 'w+', encoding='utf-8') as f:
        for line in test_raw:
            if not line:
                f.write('\n')
                continue
            sens, rec = preprocess(line)
            res, idx = [], 0
            le, ri = 0, 0
            while ri < len(sens):
                if sens[ri] == UNK:
                    if sens[le: ri]:
                        res += hmm_cut(sens[le: ri])
                    le = ri + 1
                    if idx < len(rec):
                        res += [rec[idx]]
                        idx += 1
                ri += 1
            if ri == len(sens) and sens[-1] != UNK:
                res += hmm_cut(sens[le:])
            res = '  '.join(res)
            f.write(res)
            f.write('\n')
    eval(file_path)


def make_cut(cut_func, result_file, train=False):
    file_path = os.path.join(RESULTROOT, result_file)
    line_list = test_raw
    if train:
        line_list = train_raw
    with open(os.path.join(RESULTROOT, result_file), 'w+', encoding='utf-8') as f:
        for line in line_list:
            if not line:
                f.write('\n')
                continue
            sen = line
            res = cut_func(sen)
            res = '  '.join(res)
            f.write(res)
            f.write('\n')
    eval(file_path, train)


def get_result():
    pre_make_cut(hmm_cut, 'pre_test_hmm_no_chunk.txt')
    pre_make_cut(cut, 'pre_test_hmm_chunk.txt')


def make_test_file():
    with open('../datasets/raw_test.txt', 'w', encoding='utf8') as f:
        for line in test_raw:
            f.write(line + '\n')

if __name__ == '__main__':
    make_test_file()
    # get_result()
