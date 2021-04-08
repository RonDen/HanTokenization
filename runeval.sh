#!/bin/bash

echo "max-back, max_len = 3"
perl scripts/score datasets/training_vocab.txt datasets/test.txt results/max-back-result-3.txt
echo "max-back, max_len = 4"
perl scripts/score datasets/training_vocab.txt datasets/test.txt results/max-back-result-4.txt
echo "max-back, max_len = 5"
perl scripts/score datasets/training_vocab.txt datasets/test.txt results/max-back-result-5.txt

echo "max-forward, max_len = 3"
perl scripts/score datasets/training_vocab.txt datasets/test.txt results/max-forward-result-3.txt
echo "max-forward, max_len = 4"
perl scripts/score datasets/training_vocab.txt datasets/test.txt results/max-forward-result-4.txt
echo "max-forward, max_len = 5"
perl scripts/score datasets/training_vocab.txt datasets/test.txt results/max-forward-result-5.txt
