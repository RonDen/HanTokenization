### 结巴分词

```
RECALL: 0.787
PRECISION:      0.853
F1 :    0.818

OOV Rate:       0.058
OOV Recall:     0.583
IV Recall:      0.799
```


### 最大后向匹配

```txt
RECALL: 0.903
PRECISION:      0.890
F1 :    0.896

OOV Rate:       0.058
OOV Recall:     0.000
IV Recall:      0.958
```

### Bi-LSTM

```txt
RECALL: 0.921
PRECISION:      0.927
F1 :    0.924

OOV Rate:       0.058
OOV Recall:     0.597
IV Recall:      0.940

# + CRF
RECALL: 0.923
PRECISION:      0.928
F1 :    0.926

OOV Rate:       0.058
OOV Recall:     0.629
IV Recall:      0.941

# new + merge (test_result1)
RECALL: 0.937
PRECISION:      0.942
F1 :    0.940

OOV Rate:       0.058
OOV Recall:     0.775
IV Recall:      0.947

# new + merge + random
RECALL: 0.937
PRECISION:      0.940
F1 :    0.938

OOV Rate:       0.058
OOV Recall:     0.765
IV Recall:      0.947
```

### Transformer
```txt
RECALL: 0.646
PRECISION:      0.641
F1 :    0.644

OOV Rate:       0.058
OOV Recall:     0.409
IV Recall:      0.660
```