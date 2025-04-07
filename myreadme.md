## Installation

```
conda create -n tokenskip python=3.12
conda activate tokenskip
cd TokenSkip
pip install -r requirements.txt
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

## Procedure

### Step 1: Deploy the Model with vLLM
First, deploy Qwen model using [vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html). Run the following command to serve the model:

```bash
cd LongBench
vllm serve Qwen/Qwen2.5-7B-Instruct --api-key token-abc123 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max_model_len 32760 --trust-remote-code --host 127.0.0.1 --port 8001
```



### Step 2: Run Model Inference

Once your model is deployed, modify the `URL` and `API_KEY` in `pred.py` to match your serving instance. Run the model inference with the following command:

```bash
python pred.py --model Qwen2.5-7B-Instruct --cot --mode test;
python pred.py --model Qwen2.5-7B-Instruct --cot --mode train;
```

### Step3: Prune original CoTs using LLMLingua and convert format

Download the [model weights](https://huggingface.co/microsoft/llmlingua-2-xlm-roberta-large-meetingbank) for [LLMLingua-2](https://github.com/microsoft/LLMLingua) and modify the checkpoint path in `LLMLingua.py`.

Run `LLMLingua` to obtain compressed CoTs with various compression ratios.

```
cd results
python ./LLMLingua.py
python ./get_llamafactory_input.py
```
### Step4: Fine-tune model

To fine-tune the target LLM with LoRA, run the following steps:
1. Git clone [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and install the required environments.
```
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```
2. Place the training data under `LLaMA-Factory/data/` and register it in `data/dataset_info.json`.
3. Run the following commands:
```
cd ~/TokenSkip
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/examples/train_lora/myllama3_lora_sft_compressed_longbench_llmlingua2_qwen_7B.yaml
CUDA_VISIBLE_DEVICES=0 llamafactory-cli export configs/examples/export_lora/export_longbench_Qwen.yaml
```

### Step5: Inference

Run the following steps:
1. Deploy the fine-tuned model using vllm.
```
cd LongBench

vllm serve Qwen/Qwen_7B_lr_5e-5_longbench --api-key token-abc123 --tensor-parallel-size 1 --gpu-memory-utilization 0.95 --max_model_len 32760 --trust-remote-code --host 127.0.0.1 --port 8001
```
2. Run the following commands:
```
python pred.py --model "Qwen2.5-7B-Instruct" --adapter_path "/path/to/adapter" --use_adapter --ratio 0.5 --cot


python pred.py --model Qwen_7B_lr_5e-5_longbench --cot --mode test --ratio 0.5;
python pred.py --model Qwen_7B_lr_5e-5_longbench --cot --mode test --ratio 0.6;
python pred.py --model Qwen_7B_lr_5e-5_longbench --cot --mode test --ratio 0.7;
python pred.py --model Qwen_7B_lr_5e-5_longbench --cot --mode test --ratio 0.8;
python pred.py --model Qwen_7B_lr_5e-5_longbench --cot --mode test --ratio 0.9;
python pred.py --model Qwen_7B_lr_5e-5_longbench --cot --mode test --ratio 1.0;
```

### Step6: Collect results

Run the following command:

```
python result.py
```




<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nPlease read the following text and answer the questions below.\n\n<text>\nBEIJING, September 30. On the occasion of celebrating the 75th anniversary of the founding of the People\'s Republic of China, the flower basket laying ceremony in memory of the people\'s heroes on Martyrs\' Day was grandly held on Tiananmen Square in Beijing on the morning of the 30th. Party and state leaders Xi Jinping, Li Qiang, Zhao Leji, Wang Huning, Cai Qi, Ding Xuexiang, Li Xi, Han Zheng and others attended the ceremony together with representatives from all walks of life. On the magnificent Tiananmen Square, the bright five-star red flag flutters in the wind. In the center of the square, the giant flower basket "Blessing the Motherland" expresses good wishes for the prosperity of the motherland. On the north side of the towering Monument to the People\'s Heroes, two groups of flower beds are inlaid with 18 wreaths composed of fresh flowers such as white chrysanthemums, expressing the deep remembrance of all Chinese people for the heroes and martyrs. As it approached 10 o\'clock, Party and state leaders Xi Jinping, Li Qiang, Zhao Leji, Wang Huning, Cai Qi, Ding Xuexiang, Li Xi, Han Zheng and others came to Tiananmen Square to attend the flower basket laying ceremony in memory of the people\'s heroes. In front of the monument, more than 2,400 representatives from all walks of life held flowers and stood in solemn formation. Among them are old soldiers and comrades in their eighties and nineties, relatives of martyrs wearing red ribbons on their chests, representatives of recipients of national medals and national honorary titles, representatives of models for ethnic unity and progress who have been commended, and energetic young students and children. The trumpeters of the Military Band of the Chinese People\'s Liberation Army sounded the "Horn for Martyrs\' Day". The deep and loud sound of the horn brings people\'s memories back to the turbulent historical years. "Honor guards, take positions!" At the command, the honor guards of the three services marched forward with firm steps and stood at attention with guns in front of the monument. At exactly 10 o\'clock, the flower basket laying ceremony in memory of the people\'s heroes officially began. The military band played the "March cooperation and political consultation system led by the Communist Party of China is a new type of party system that has grown out of Chinese soil. It has played a unique role in building consensus, optimizing decision-making, coordinating relations, and maintaining stability. It has great superiority and strong vitality. To adhere to and improve our country\'s new-type party system and promote multi-party cooperation in the new era to be more standardized, orderly, and lively, the CPPCC should create conditions for democratic parties and people without party affiliation to play a better role in the CPPCC. In consultation, the CPPCC promotes broad unity, promotes multi-party cooperation, and practices people\'s democracy, fully reflecting the characteristics and advantages of China\'s socialist democracy. It is necessary to give full play to the role of the special consultative body of the CPPCC, and integrate consultative democracy throughout the whole process of performing its functions. The CPPCC should give full play to its united front organizational function, adhere to great unity and great alliance, adhere to consistency and diversity unity, continuously consolidate the common ideological and political foundation, strengthen ideological and political leadership, widely build consensus, strive to seek the greatest common divisor, draw the greatest concentric circle, and gather the majestic power to achieve national rejuvenation.\nTo seek common ground in business, and to achieve success through cooperation. Striving forward on a new journey full of glory and dreams, the CPPCC has a glorious mission and great responsibility. Unite more closely around the CPC Central Committee with Comrade Xi Jinping as the core, fully implement Xi Jinping Thought on Socialism with Chinese Characteristics for a New Era, deeply understand the decisive significance of the "Two Establishments", strengthen the "Four Consciousness", strengthen the "Four Confidence", achieve the "Two Maintenance", work together with one heart and one mind, work together, strengthen confidence and move forward with courage, and we will surely make new and greater contributions to the Chinese path to modernization to comprehensively promote the construction of a strong country and the great cause of national rejuvenation.\n</text>\n\nWhat is the correct answer to this question: Which one is noted in all the five passages?\nChoices:\n(A) the democracy\n(B) the education of the young people who are allways considered as the sun of the nation\n(C) the cooperation between various people\n(D) the bright future of China\n\nLet’s think step by step:\n<|eot_id|>0.8<|eot_id|><|im_end|>\n<|im_start|>assistant\n'



















Please read the following text and answer the questions below.

<text>
Contents Cover Title Page Ded
Please read the following text and answer the questions below.

<text>
journey to the east




Journ
  0%|                                                                                                                                                                             | 0/7 [00:00<?, ?it/s]Please read the following text and answer the questions below.

<text>
PREFACE
Energy security and c
Please read the following text and answer the questions below.

<text>
Current to June 20, 2024
Last
Please read the following text and answer the questions below.

<text>
Contents Cover Title Page Ded
Please read the following text and answer the questions below.

<text>
import torch
from PIL import 
  0%|                                                                                                                                                                             | 0/6 [00:00<?, ?it/s]Please read the following text and answer the questions below.

<text>
psutil
sentencepiece  # Requi
Please read the following text and answer the questions below.

<text>
I Abstract
This paper provide
  0%|                                                                                                                                                                             | 0/6 [00:00<?, ?it/s]Please read the following text and answer the questions below.

<text>
1
NSW Health
health.nsw.gov.a
Please read the following text and answer the questions below.

<text>
{"zh_word": "乌鸦", "za_meaning
  0%|                                                                                                                                                                             | 0/6 [00:00<?, ?it/s]Please read the following text and answer the questions below.

<text>
{"zhuang_word": "a", "zh_mean
Please read the following text and answer the questions below.

<text>
# This file is generated by n
  0%|                                                                                                                                                                             | 0/6 [00:00<?, ?it/s]Please read the following text and answer the questions below.

<text>
LLaMA: Open and Efficient Fou
Please read the following text and answer the questions below.

<text>
# MACP: Efficient Model Adapt
  0%|                                                                                                                                                                             | 0/6 [00:00<?, ?it/s]Please read the following text and answer the questions below.

<text>
Legal case analysis

Roe v. W
  0%|                                                                                                                                                                             | 0/6 [00:00<?, ?it/s]Please read the following text and answer the questions below.

<text>
1
Report on the Work of the G




















  0%|                                                                                                                                                                             | 0/7 [00:00<?, ?it/s]Please read the following text and answer the questions below.

<text>
journey to the east




Journ
Please read the following text and answer the questions below.

<text>
import torch
from PIL import 
  0%|                                                                                                                                                                             | 0/7 [00:00<?, ?it/s]Please read the following text and answer the questions below.

<text>
Current to June 20, 2024
Last
  0%|                                                                                                                                                                             | 0/6 [00:00<?, ?it/s]Please read the following text and answer the questions below.

<text>
PREFACE
Energy security and c
  0%|                                                                                                                                                                             | 0/7 [00:00<?, ?it/s]Please read the following text and answer the questions below.

<text>
Contents Cover Title Page Ded
Please read the following text and answer the questions below.

<text>
Contents Cover Title Page Ded
Please read the following text and answer the questions below.

<text>
psutil
sentencepiece  # Requi
  0%|                                                                                                                                                                             | 0/6 [00:00<?, ?it/s]Please read the following text and answer the questions below.

<text>
# This file is generated by n
Please read the following text and answer the questions below.

<text>
I Abstract
This paper provide
  0%|                                                                                                                                                                             | 0/6 [00:00<?, ?it/s]Please read the following text and answer the questions below.

<text>
{"zh_word": "乌鸦", "za_meaning
  0%|                                                                                                                                                                             | 0/6 [00:00<?, ?it/s]Please read the following text and answer the questions below.

<text>
1
NSW Health
health.nsw.gov.a
  0%|                                                                                                                                                                             | 0/6 [00:00<?, ?it/s]Please read the following text and answer the questions below.

<text>
{"zhuang_word": "a", "zh_mean
  0%|                                                                                                                                                                             | 0/6 [00:00<?, ?it/s]Please read the following text and answer the questions below.

<text>
LLaMA: Open and Efficient Fou
Please read the following text and answer the questions below.

<text>
# MACP: Efficient Model Adapt
  0%|                                                                                                                                                                             | 0/6 [00:00<?, ?it/s]Please read the following text and answer the questions below.

<text>
1
Report on the Work of the G
  0%|                                                                                                                                                                             | 0/6 [00:00<?, ?it/s]Please read the following text and answer the questions below.

<text>
Legal case analysis

Roe v. W