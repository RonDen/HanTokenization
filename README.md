# HAN中文分词-2021 Spring NLP Homework

### 训练样本词频统计

使用`collection.Counter`和`nltk`相应工具包完成训练集词频统计分析。展示出现次数最多的前80个词。

![词频](notebooks\前80个词的词频.png)

![词频-累加](notebooks\前80个词的词频-累加.png)


### 结巴分词baseline

使用结巴分词默认配置（`jieba.cut`）得到结果于文件[jieba-test-result.txt](results\jieba-test-result.txt)中，执行测试脚本。

```bash
perl scripts/score datasets/training_vocab.txt datasets/test.txt results/jieba-test-result.txt
# 或者直接运行runeval.sh
```
