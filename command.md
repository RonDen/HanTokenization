### 一些命令

- #### Bi-LSTM

  ```
  # punc_flag should be False
  CUDA_VISIBLE_DEVICES=6,7,8,9 python3 train.py

  # With punc. punc_flag = True
  CUDA_VISIBLE_DEVICES=6,7,8,9 nohup python3 -u train.py \
	--vocab_path ../pretrained_models/Glove/vocab_with_punc.txt \
	--pretrained_embedding_path ../pretrained_models/Glove/glove_embed_with_punc.txt \
  	--last_model_path ../models/bilstm_with_punc/last/model.bin \
  	--best_model_path ../models/bilstm_with_punc/best/model.bin \
  	--result_path ../results/bilstm_with_punc/test_result.txt \
  	--model_type bilstm > ../logs/bilstm_with_punc_training.out 2>&1 &!
	
  # Update punc. punc_flag = False
  CUDA_VISIBLE_DEVICES=6,7,8,9 nohup python3 -u train.py \
  	--last_model_path ../models/bilstm_new/last/model.bin \
  	--best_model_path ../models/bilstm_new/best/model.bin \
  	--result_path ../results/bilstm_new/test_result.txt \
  	--model_type bilstm > ../logs/bilstm_new_training.out 2>&1 &!
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

  # With punc. punc_flag = True
  CUDA_VISIBLE_DEVICES=6,7,8,9 nohup python3 -u train.py \
	--vocab_path ../pretrained_models/Glove/vocab_with_punc.txt \
	--pretrained_embedding_path ../pretrained_models/Glove/glove_embed_with_punc.txt \
  	--last_model_path ../models/bilstmcrf_with_punc/last/model.bin \
  	--best_model_path ../models/bilstmcrf_with_punc/best/model.bin \
  	--result_path ../results/bilstmcrf_with_punc/test_result.txt \
  	--model_type bilstmcrf > ../logs/bilstmcrf_with_punc_training.out 2>&1 &!
  ```

  