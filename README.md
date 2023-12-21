# Hippo: radiology-GPT

<img width="511" alt="스크린샷 2023-12-21 오후 5 03 52" src="https://github.com/yjeong-k/radiology-GPT/assets/75728717/eb07ab82-45e7-4d45-9a52-e270fb203148" width="720" height="800" >

## Introduction
Hippo(radiology-GPT)는 방사선 판독보고서 데이터로 파인튜닝한 의료 도메인의 챗봇입니다.  
Hippo라는 이름은 의학의 아버지 Hippocrates의 이름에서 따 온 것입니다.  


* Base Model: [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)  
* Dataset: [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) 약 160k개의 노트를 사용하였습니다.  
* Method: Instruction-following(by Stanford Alpaca) 방식으로 학습을 진행하였습니다. 데이터의 생성에는 GPT-3.5 turbo API를 이용하였습니다.


## How to Use
### Environment
제공드린 Dockerfile을 사용하시면 됩니다.  

* Docker Image Build
```bash
docker build -t hippo:latest .
```

* Docker Run Container
```bash
docker run -v MOUNT_PATH:/workspace --gpus GPU_NUM -it --name "hippo" hippo:latest
```
-v 옵션을 지정하여 볼륨을 마운트하였습니다. MOUNT_PATH는 컨테이너에 마운트할 로컬 경로를 의미합니다.  
--gpus 옵션을 지정하여 사용할 GPU를 지정할 수 있습니다.  
-it 옵션을 지정하여 터미널을 이용하여 컨테이너와 상호작용할 수 있습니다.  
"hippo"는 컨테이너의 이름, hippo:latest는 이미지 이름입니다.  

* Container 재사용
실행중인 컨테이너에 재진입하여 작업하는 경우, 다음의 명령어를 사용하시면 됩니다.
```bash
docker exec -it hippo /bin/bash
```
hippo는 실행중인 컨테이너의 이름입니다.

### Data Preprocessing

MIMIC-CXR 데이터셋에서 방사선 판독보고서 파일인 notes를 전처리합니다.  
보고서마다 형식이 제각각이기 때문에, 보고서에서 핵심 정보를 담고 있는  

** "EXAMINATION", "HISTORY", "INDICATION", "TECHNIQUE",  
"COMPARISON", "FINDINGS", "IMPRESSION"**  

항목을 중심으로 전처리를 수행하였습니다.

```bash
python preprocessing/preprocess_mimic_cxr --input_path INPUT_PATH --save_path SAVE_PATH
```
* input_path: MIMIC-CXR notes 데이터셋이 위치한 경로입니다.  
* save_path: 전처리된 데이터셋이 저장될 경로입니다.

### Data Generation

Data Generation은 다음의 단계를 거쳐 이루어집니다.  

1. OpenAI API를 이용하여 instruction을 생성합니다.  
```bash
python preprocessing/instruction_generator.py --input_path INPUT_PATH --save_path SAVE_PATH --api_key API_KEY
```  
이 때 max_requesets/token_per_minute, max_attemps 등 API 세부 설정을 변경하실 수 있습니다.  
세부 파라미터는 코드를 참조하세요!  
  
2. API Response에서 생성된 Instruction을 후처리합니다.  
```bash
python preprocessing/postproc_question.py --input_path INPUT_PATH --save_path SAVE_PATH
```  
  
3. OpenAI API를 다시 이용하여 후처리한 데이터에 대한 Answer를 생성합니다.  
```bash
python preprocessing/answer_generator.py --input_path INPUT_PATH --save_path SAVE_PATH --api_key API_KEY
```  
4. 생성한 Instruction-Answer 쌍을 후처리합니다.  
```bash
python preprocessing/answer_postprocess.py --input_path INPUT_PATH --save_path SAVE_PATH
```  


### Fine Tuning

Huggingface에서 Llama 모델을 사용할 때 CLI login을 해야 합니다.  
Huggingface 계정에서 token을 발급받아 사용하시면 됩니다.  
이 때 명령어는 다음과 같습니다.  
```bash
huggingface-cli login
YOUR_HF_TOKEN
n ## git credential
```

fine tuning에는 다음의 명령어를 사용합니다. 파라미터의 값은 조정하시면 됩니다.  

```bash
python train/fine_tuning.py --model_name_or_path "meta-llama/Llama-2-7b-chat-hf" --data_path TRAINING_DATA_PATH --output_dir CKPT_OUTPUT_PATH --num_train_epochs 3 --per_device_train_batch_size 4 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8 --evaluation_strategy "no" --save_strategy "epoch" --learning_rate 2e-4 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --model_max_length 4096 --gradient_checkpointing True --ddp_timeout 18000
```

### Inference
학습된 모델을 이용하여 답변을 생성하고자 하는 경우, 다음의 명령어를 사용하시면 됩니다.  

```bash
python inference/inference.py --ft_path CKPT_PATH
```

해당 모듈에서는 학습된 radiology_GPT가 챗봇 형식으로 사용자와 질의응답을 하게 됩니다.  
이전에 이루어졌던 대화를 반영하여 답변을 생성하게 됩니다.  

### Comparison
다른 모델들의 답변을 받아 보고 싶으실 경우, Comparison 디렉토리에 있는 모듈들을 활용하시면 됩니다.  


### Evaluation
Hippo는 1. Accuracy 2. Conciseness 3. Understandability의 3가지 지표에 기반하여, GPT-4가 1~4점 척도로 점수를 매겨 평가를 수행합니다.  
GPT-4로부터 각 모델의 답변을 평가하는 경우, Evaluation 내에 있는 모듈들을 활용하시면 됩니다.


### Demo
Hippo의 데모 버전은 demo.py 모듈에 구현되어 있습니다.  
Gradio 라이브러리를 사용하여 사용자가 채팅 인터페이스로 Hippo를 사용해볼 수 있습니다.  
(Huggingface chat demo 코드 참조)  
<img width="1283" alt="스크린샷 2023-12-21 오후 5 41 41" src="https://github.com/yjeong-k/radiology-GPT/assets/75728717/434edb9b-d243-491e-b040-96d28b97b0c3">

  
Demo 실행 명령어는 다음과 같습니다.
```bash
python demo/demo.py
```  
  
터미널에 출력되는 링크를 클릭하시면, 다음의 인터페이스에서 Hippo를 사용하실 수 있습니다.  


# Reference
[KAIST Asclepius](https://github.com/starmpcc/Asclepius)  
[Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)  
[Open AI](https://github.com/openai/openai-cookbook/tree/main)  
[Huggingface Llama2 chat demo](https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat/blob/main/app.py)  
