import os
import re
import time

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
test_set_split = [line.split('  ') for line in test_set]

train_raw = [''.join(line) for line in train_set_split]
test_raw = [''.join(line) for line in test_set_split]


def timer(func):
    def wrapper(*argc, **argv):
        st = time.time()
        res = func(*argc, **argv)
        print("time spend: %.3f\n" % (time.time() - st))
        return res
    return wrapper


# @timer
def max_forward(line: str, max_len = 4):
    """
    最大前向匹配，从前向后扫描句子，从大到小缩小长度。
    """
    res = []
    n = len(line)
    idx = 0
    while idx < n:
        lens = max_len
        while lens > 0:
            sub = line[idx: idx + lens]
            # print(sub)
            if sub in vocab:
                res.append(sub)
                idx += lens
                lens = max_len
            else:
                lens -= 1
                if lens == 0:
                    idx += 1
    return res


# @timer
def max_back(line: str, max_len = 4):
    """
    最大后向匹配，从后向前扫描句子，从大到小缩小长度。
    """
    res = []
    n = len(line)
    idx = n
    while idx >= 1:
        lens = max_len
        while lens > 0:
            lens = min(lens, idx)
            sub = line[idx-lens: idx]
            if sub in vocab:
                res.append(sub)
                idx -= lens
                lens = max_len
            else:
                lens -= 1
                if lens == 0:
                    idx -= 1

    return list(reversed(res))

def show5():
    for i in range(5):
        print(test_raw[i])
        print(max_back(test_raw[i]))

# 查看前5个的效果
# show5()

def eval(file_path, train=False, report_file="", help_line=""):
    gold = 'training' if train else 'test'
    cmd = 'perl /home/luod/class/nlp/HanTokenization/scripts/score /home/luod/class/nlp/HanTokenization/datasets/training_vocab.txt /home/luod/class/nlp/HanTokenization/datasets/{}.txt {}'.format(gold, file_path)
    if report_file:
        os.system('echo {} >> {}'.format(help_line+'\n', report_file))
        cmd += ' | tee -a {}'.format(report_file)
    os.system(cmd)


def run_back_seg(max_len: int):
    filename = 'max-back-result.txt'
    report_file = os.path.join('/home/luod/class/nlp/HanTokenization/logs/', 'max-back-report.txt')
    help_line = 'seg with max len = %d' % max_len
    with open(os.path.join(RESULTROOT, filename), 'w+', encoding='utf-8') as f:
        st = time.time()
        result = '\n'.join(['  '.join(max_back(line, max_len=max_len)) for line in test_raw])
        print("Spent %.3fs\n" % (time.time() - st))
        f.write(result)
    eval(os.path.join(RESULTROOT, filename), train=False, report_file=report_file, help_line=help_line)


def run_forward_seg(max_len: int):
    filename = 'max-forward-result.txt'
    report_file = os.path.join('/home/luod/class/nlp/HanTokenization/logs/', 'max-forward-report.txt')
    help_line = 'seg with max len = %d' % max_len
    with open(os.path.join(RESULTROOT, filename), 'w+', encoding='utf-8') as f:
        st = time.time()
        result = '\n'.join(['  '.join(max_forward(line, max_len=max_len)) for line in test_raw])
        print("Spent %.3fs\n" % (time.time() - st))
        f.write(result)
    eval(os.path.join(RESULTROOT, filename), train=False, report_file=report_file, help_line=help_line)


def run_batch():
    for i in range(1, 6):
        run_back_seg(i)


run_batch()
# run_back_seg(4)
