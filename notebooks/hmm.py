import os
import json
from collections import Counter


DATAROOT = '../datasets'
RESULTROOT = '../results'
VOCAB_FILE = os.path.join(DATAROOT, 'training_vocab.txt')
TRAIN_FILE = os.path.join(DATAROOT, 'training.txt')
TEST_FILE = os.path.join(DATAROOT, 'test.txt')
PRE_TRAIN = os.path.join(DATAROOT, 'training_pre.json')
PRE_TEST = os.path.join(DATAROOT, 'test_pre.json')

train_set = []

class Item:
    def __init__(self, sentence, sentence_id, label) -> None:    
        self.sentence = sentence
        self.sentence_id = sentence_id
        self.label = label
    
    def __repr__(self) -> str:
        return '\n'.join([self.sentence, str(self.sentence_id)[1:-1], str(self.label)[1:-1]])

    def __str__(self) -> str:
        return '\n'.join([self.sentence, str(self.sentence_id)[1:-1], str(self.label)[1:-1]])


with open(PRE_TRAIN, 'r', encoding='utf8') as f:
    train_set = [Item(**json.loads(line)) for line in f.readlines()]

# print(train_set[0])
# print(train_set[1])

word_counter = Counter()
state_counter = Counter()
out_counter = Counter()

# [(1, 176175), (104, 54650), (105, 17688), (106, 17454), (107, 13582), (108, 12846), (109, 12513), (110, 12320), (111, 12257), (112, 11848)]

for item in train_set:
    word_counter.update(item.sentence)
    state_counter.update(item.label)
    out_counter.update(item.sentence_id)

# [('，', 74143), ('的', 54650), ('。', 35604), ('、', 22979), ('国', 17688), ('一', 17454), ('在', 13582), ('中', 12846), ('人', 12513), ('了', 12320)]
print(word_counter.most_common(n=10))
# [(0, 585232), (2, 585232), (3, 524715), (1, 131296)]
# print(state_counter.most_common())
# [(1, 176175), (104, 54650), (105, 17688), (106, 17454), (107, 13582), (108, 12846), (109, 12513), (110, 12320), (111, 12257), (112, 11848)]
print(out_counter.most_common(n=10))
