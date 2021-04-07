import os
import jieba

DATAROOT = '../datasets'
RESULTROOT = '../results'
VOCAB_FILE = os.path.join(DATAROOT, 'training_vocab.txt')
VOCAB_FREQ = os.path.join(RESULTROOT, 'vocab-freq.txt')
TRAIN_FILE = os.path.join(DATAROOT, 'training.txt')
TEST_FILE = os.path.join(DATAROOT, 'test.txt')

RESULT_FILE_TRAIN = os.path.join(RESULTROOT, 'jieba-train-result.txt')
RESULT_FILE_TEST = os.path.join(RESULTROOT, 'jieba-test-result.txt')
RESULT_FILE_TEST_IM = os.path.join(RESULTROOT, 'jieba-test-result-im.txt')

vocab = set()
train_set = []
test_set = []

with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
    vocab = set(map(str.strip, f.readlines()))

with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
    train_set = list(map(str.strip, f.readlines()))

with open(TEST_FILE, 'r', encoding='utf-8') as f:
    test_set = list(map(str.strip, f.readlines()))


# print('vocab size ', len(vocab))
# print('training set size: ', len(train_set))
# print('test set size: ', len(test_set))
# print("ten item of training set")
# print(train_set[:5])
# print("ten item of test set")
# print(test_set[:5])

jieba.set_dictionary(VOCAB_FREQ)

train_set_split = [line.split('  ') for line in train_set]
test_set_split = [line.split('  ') for line in test_set]

train_raw = [''.join(line) for line in train_set_split]
test_raw = [''.join(line) for line in test_set_split]

# for line in train_raw[:5]:
#     print('  '.join(jieba.cut(line)))

def write_test_result():
    with open(RESULT_FILE_TEST_IM, 'w+', encoding='utf-8') as f:
        f.writelines(['  '.join(jieba.cut(line)) + '\n' for line in test_raw])

write_test_result()

