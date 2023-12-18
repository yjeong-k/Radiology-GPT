# radiology-GPT

## Introduction
- dataset: Mimic-CXR
- license: 

## How to Use
### Data Preprocessing

### Data Generation

### Fine Tuning

Huggingface에서 Llama 모델을 사용할 때 CLI login을 해야 합니다.
Huggingface 계정에서 token을 발급받아 사용하시면 됩니다.
이 때 명령어는 다음과 같습니다. 
```sh
huggingface-cli login
YOUR_HF_TOKEN
n
```

```sh
python FINE_TUNING_MODULE_PATH --model_name_or_path "meta-llama/Llama-2-7b-chat-hf" --data_path TRAINING_DATA_PATH(json) --output_dir CKPT_OUTPUT_PATH --num_train_epochs 2 --per_device_train_batch_size 4 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8 --evaluation_strategy "no" --save_strategy "epoch" --learning_rate 2e-4 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --model_max_length 4096 --gradient_checkpointing True --ddp_timeout 18000
```
### Inference
```sh
python INFERENCE_MODULE_PATH --ft_path CKPT_PATH
```


# Citation
