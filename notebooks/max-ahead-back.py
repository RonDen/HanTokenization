import os

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


def max_back(line: str, max_len = 4):
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

def show5():
    for i in range(5):
        print(test_raw[i])
        print(max_back(test_raw[i]))

# 查看前5个的效果
# show5()
# with open(os.path.join(RESULTROOT, 'max-back-result.txt'), 'w+', encoding='utf-8') as f:

#     result = '\n'.join(['  '.join(max_back(line)) for line in test_raw])
#     f.write(result)

