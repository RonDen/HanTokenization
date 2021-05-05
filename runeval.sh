#!/bin/bash

# echo "jieba without dictionary eval result"
perl scripts/score datasets/training_vocab.txt datasets/test.txt /home/luod/class/nlp/HanTokenization/results/jieba-test-result-hmm.txt

# echo "jieba with dictionary eval result"
# perl scripts/score datasets/training_vocab.txt datasets/test.txt results/jieba-test-result-im.txt

# echo "jieba with dictionary no hmm eval result"
# perl scripts/score datasets/training_vocab.txt datasets/test.txt results/jieba-test-result-no-hmm.txt

# echo "jieba with dictionary paddle no hmm eval result"
# perl scripts/score datasets/training_vocab.txt datasets/test.txt results/jieba-test-result-paddle-no-hmm.txt

# echo "jieba with dictionary paddle with hmm eval result"
# perl scripts/score datasets/training_vocab.txt datasets/test.txt results/jieba-test-result-paddle-hmm.txt
# echo "max-back, max_len = 3"
# perl scripts/score datasets/training_vocab.txt datasets/test.txt results/max-back-result-3.txt
# echo "max-back, max_len = 4"
# perl scripts/score datasets/training_vocab.txt datasets/test.txt results/max-back-result-4.txt
# echo "max-back, max_len = 5"
# perl scripts/score datasets/training_vocab.txt datasets/test.txt results/max-back-result-5.txt

# echo "max-forward, max_len = 3"
# perl scripts/score datasets/training_vocab.txt datasets/test.txt results/max-forward-result-3.txt
# echo "max-forward, max_len = 4"
# perl scripts/score datasets/training_vocab.txt datasets/test.txt results/max-forward-result-4.txt
# echo "max-forward, max_len = 5"
# perl scripts/score datasets/training_vocab.txt datasets/test.txt results/max-forward-result-5.txt
