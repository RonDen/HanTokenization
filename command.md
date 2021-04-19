### 一些命令

- #### Bi-LSTM

  ```
  CUDA_VISIBLE_DEVICES=6,7,8,9 python3 train.py
  ```

- #### Bi-LSTM+CRF

  ```
  CUDA_VISIBLE_DEVICES=6,7,8,9 python3 train.py \
  	--last_model_path ../models/bilstmcrf/last/model.bin \
  	--best_model_path ../models/bilstmcrf/best/model.bin \
  	--result_path ../results/bilstmcrf/test_result.txt \
  	--model_type bilstmcrf
  	
  CUDA_VISIBLE_DEVICES=6,7,8,9 nohup python3 -u train.py \
  	--last_model_path ../models/bilstmcrf/last/model.bin \
  	--best_model_path ../models/bilstmcrf/best/model.bin \
  	--result_path ../results/bilstmcrf/test_result.txt \
  	--model_type bilstmcrf > ../logs/bilstmcrf_training.out 2>&1 &!
  ```

  