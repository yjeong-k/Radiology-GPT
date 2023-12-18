# radiology-GPT

## Introduction

방사선 판독보고서 데이터로 파인튜닝한 의료 도메인의 챗봇입니다.

- [base model: Llama-2-7b-chat-hf] (https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).
- [dataset: MIMIC-CXR] (https://physionet.org/content/mimic-cxr/2.0.0/).
- License: 

## How to Use
### Environment
Dockerfile을 사용하시면 됩니다.

- Docker Image Build
```sh
docker build -t hippo:latest .
```

- Docker Container 실행
```sh
docker docker run -v MOUNT_PATH:/workspace --gpus GPU_NUM -it --name "hippo" hippo:latest
```

### Data Preprocessing

### Data Generation

### Fine Tuning

Huggingface에서 Llama 모델을 사용할 때 CLI login을 해야 합니다.
Huggingface 계정에서 token을 발급받아 사용하시면 됩니다.
이 때 명령어는 다음과 같습니다. 
```sh
$ huggingface-cli login
$ YOUR_HF_TOKEN
$ n ## git credential
```

fine tuning에는 다음의 명령어를 사용합니다. 파라미터의 값은 조정하시면 됩니다.

```sh
python fine_tuning.py --model_name_or_path "meta-llama/Llama-2-7b-chat-hf" --data_path TRAINING_DATA_PATH --output_dir CKPT_OUTPUT_PATH --num_train_epochs 3 --per_device_train_batch_size 4 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8 --evaluation_strategy "no" --save_strategy "epoch" --learning_rate 2e-4 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --model_max_length 4096 --gradient_checkpointing True --ddp_timeout 18000
```

### Inference
학습된 모델을 이용하여 답변을 생성하고자 하는 경우, 다음의 명령어를 사용하시면 됩니다.

```sh
python inference.py --ft_path CKPT_PATH
```

해당 모듈에서는 학습된 radiology_GPT가 챗봇 형식으로 사용자와 질의응답을 하게 됩니다.
이전에 이루어졌던 대화를 반영하여 답변을 생성하게 됩니다.


# Citation

# Code
[KAIST Asclepius](https://github.com/starmpcc/Asclepius)  
[Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)  
[Open AI](https://github.com/openai/openai-cookbook/tree/main)  
[Huggingface Llama2 chat demo](https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat/blob/main/app.py)  