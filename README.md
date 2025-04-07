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