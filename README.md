# Hippo: radiology-GPT

<img width="511" alt="스크린샷 2023-12-21 오후 5 03 52" src="https://github.com/yjeong-k/radiology-GPT/assets/75728717/eb07ab82-45e7-4d45-9a52-e270fb203148">

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
  
2. API Response에서 생성된 Instruction을 후처리하여, 각 Instruction에 대한 answer를 생성하도록 명령하는 prompt를 생성합니다.
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

fine tuning에는 다음의 명령어를 사용합니다.

```bash
python train/fine_tuning.py --model_name_or_path "meta-llama/Llama-2-7b-chat-hf" --data_path TRAINING_DATA_PATH --output_dir CKPT_OUTPUT_PATH --num_train_epochs 3 --per_device_train_batch_size 4 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8 --evaluation_strategy "no" --save_strategy "epoch" --learning_rate 2e-4 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --model_max_length 4096 --gradient_checkpointing True --ddp_timeout 18000
```
  
  
하이퍼파라미터는 조정하시면 됩니다.  
각 하이퍼파라미터의 자세한 설명은 [hyperparameter description](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)을 참고해주세요!


### Inference
학습된 모델을 이용하여 답변을 생성하고자 하는 경우, 다음의 명령어를 사용하시면 됩니다.  

```bash
python inference/inference.py --ft_path CKPT_PATH
```

해당 모듈에서는 학습된 radiology_GPT가 챗봇 형식으로 사용자와 질의응답을 하게 됩니다.  
이전에 이루어졌던 대화를 반영하여 답변을 생성하게 됩니다.  

### Comparison
다른 모델들의 답변을 받아 보고 싶으실 경우, Comparison 디렉토리에 있는 모듈들을 활용하시면 됩니다.  

아래의 모든 모듈들은 argument로 input_path와 save_path를 지정해주셔야 합니다.  

1) bard.py  
바드의 답변을 받아오고 싶을 때 사용하시면 됩니다. 바드에 전송할 prompt를 'prompt'라는 column에 담고있는 csv 파일을 input_path에 명시해주시면, 'bard_answer'이라는 새로운 column에 답변을 저장하여 save_path에 csv 파일로 반환합니다.  
위 파일을 사용하실 때는 bard_secrets.py라는 파일이 추가적으로 필요합니다. 해당 파일은 bard.py와 동일한 directory hierarchy에 위치해두시면 되며, 아래와 같은 내용을 담고 있습니다.
```bash
COOKIE_DICT = {
    "__Secure-1PSID": "yours",
    "__Secure-1PSIDTS": "yours",
}
```
크롬에서 바드에 접속하신 후, F12 키를 눌러 개발자 모드로 진입합니다. 이후 쿠키 값 중 `__Secure-1PSID`와 `__Secure-1PSIDTS` 값을 찾아 bard_secrets.py 파일에 넣어 저장해주시면 됩니다.  
  
2) llama2.py  
라마2의 답변을 받아오고 싶을 때 사용하시면 됩니다. 라마2에 전송할 prompt를 'prompt'라는 column에 담고있는 csv 파일을 input_path에 명시해주시면, 'llama2_answer'이라는 새로운 column에 답변을 저장하여 save_path에 csv 파일로 반환합니다.  
라마2를 사용하기 위해서는 huggingface CLI login이 필요합니다. 앞서 Fine Tuning 섹션에서 설명드린 방법대로 login을 진행해주시면 됩니다.

3) medAlpaca.py  
medAlpaca(7B)의 답변을 받아오고 싶을 때 사용하시면 됩니다. medAlpaca에 전송할 prompt를 'prompt'라는 column에 담고있는 csv 파일을 input_path에 명시해주시면, 'medAlapca_answer'이라는 새로운 column에 답변을 저장하여 save_path에 csv 파일로 반환합니다.  

4) hippo.py  
본 프로젝트에서 개발한 hippo의 답변을 받아오고 싶을 때 사용하시면 됩니다. Hippo에 전송할 prompt를 'prompt'라는 column에 담고있는 csv 파일을 input_path에 명시해주시면, 'hippo_answer'이라는 새로운 column에 답변을 저장하여 save_path에 csv 파일로 반환합니다.


### Evaluation
Hippo는 1. Accuracy 2. Conciseness 3. Understandability의 3가지 지표에 기반하여, GPT-4가 1~4점 척도로 점수를 매겨 평가를 수행합니다.  
GPT-4로부터 각 모델의 답변을 평가하는 경우, Evaluation 내에 있는 모듈들을 활용하시면 됩니다.  

아래의 모든 모듈을 사용하기 위해서는 `secrets.py` 파일이 필요합니다. 해당 파일은 다음과 같은 내용을 담고 있으며, evaluate_*.py 파일들과 동일한 directory hierarchy에 위치시켜두시면 됩니다.  
```
OPENAI_API_KEY = "your API key"  ## GPT4를 사용하기 위한 OpenAI API key
```  
또한, argument로 input_path와 save_path를 명시해주셔야 합니다.  

  
1) evaluate_accuracy.py  
accuracy를 척도로 평가를 진행하고 싶을 때 사용하시면 됩니다. input csv 파일은 방사선 판독보고서를 'report'라는 열에, 그에 대한 질문을 'question'이라는 열에 담고 있어야 합니다. 또한 비교 평가에 활용될 모델들의 답변은 'modelName_answer'의 열에 저장되어 있습니다. modelName 부분은 자유롭게 지정해주시면 되지만, 끝에는 항상 _answer가 붙어있어야 합니다.
GPT4의 평가 결과로 도출된 점수들은 modelName_score라는 새로운 열에 저장되어 save_path에 csv 파일로 반환됩니다.

2) evaluate_conciseness.py  
conciseness를 척도로 평가를 진행하고 싶을 때 사용하시면 됩니다. 세부 사항은 위와 같습니다.

3) evaluate_understandability.py  
understandability를 척도로 평가를 진행하고 싶을 때 사용하시면 됩니다. 세부 사항은 위와 같습니다. 



### Demo

<img width="1283" alt="스크린샷 2023-12-21 오후 5 41 41" src="https://github.com/yjeong-k/radiology-GPT/assets/75728717/434edb9b-d243-491e-b040-96d28b97b0c3">  

demo.py 모듈에서는 사용자가 채팅 인터페이스를 통해 Hippo를 사용해 볼 수 있습니다.  
Demo는 Gradio 라이브러리를 사용하여 구현되었습니다. (Huggingface chat demo 코드 참조)  

  
Demo 실행 명령어는 다음과 같습니다.
```bash
python demo/demo.py
```  
  
명령어 실행 후 터미널에 출력되는 public url을 클릭하시면 Demo를 사용하실 수 있습니다.  

# Reference
[KAIST Asclepius](https://github.com/starmpcc/Asclepius)  
[Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)  
[Open AI](https://github.com/openai/openai-cookbook/tree/main)  
[Huggingface Llama2 chat demo](https://huggingface.co/spaces/huggingface-projects/llama-2-7b-chat/blob/main/app.py)  
