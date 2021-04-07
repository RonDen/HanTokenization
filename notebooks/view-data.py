#!/usr/bin/env python
# coding: utf-8

# In[1]:

# In[1]:


import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import Counter


# In[2]:


DATAROOT = '../datasets'
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


# In[3]:


print('vocab size ', len(vocab))
print('training set size: ', len(train_set))
print('test set size: ', len(test_set))
print("ten item of training set")
print(train_set[:5])
print("ten item of test set")
print(test_set[:5])


# In[4]:


train_set_split = [line.split('  ') for line in train_set]


# In[5]:


len(train_set) == len(train_set_split)


# In[6]:


train_set_split[:5]


# In[7]:


cnt = Counter()
for line in train_set_split:
    cnt.update(line)


# In[8]:


len(cnt) < len(vocab)


# In[9]:


cnt0 = Counter(train_set_split[0])
print(len(train_set_split[0]))
print(len(cnt0))
print(cnt0)


# In[10]:


cnt.most_common()


# In[11]:

# In[12]:


from wordcloud import WordCloud


# In[13]:


npcnt = np.array(cnt.most_common())


# In[14]:


import pandas as pd


# In[15]:


cnt_series = pd.Series(cnt.most_common())


# In[27]:


print("每个词平均出现次数：", sum(cnt.values()) / len(cnt))


# In[29]:


words, times = [], []
for word, time in cnt.most_common():
    words.append(word), times.append(time)


# In[38]:

import matplotlib
matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号

from nltk.probability import FreqDist

frqdist = FreqDist(cnt)

plt.figure(figsize=(16, 8))
plt.grid(False)
frqdist.plot(80)
plt.show()

plt.figure(figsize=(16, 8))
plt.grid(False)
frqdist.plot(80, cumulative=True)

# In[ ]:



