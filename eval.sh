BENCHMARK="gsm8k" # "gsm8k", "math"
OUPTUT_DIR="outputs/Qwen2.5-7B-Instruct/${BENCHMARK}/"
MODEL_PATH="./model/Qwen2.5-7B-Instruct"
MODEL_SIZE="7b"
MODEL_TYPE="qwen" # "llama3", "qwen"
DATA_TYPE="test" # "train", "test"

# Generation Settings
MAX_NUM_EXAMPLES=100000000000000
MAX_NEW_TOKENS=512 # 512 for gsm8k, 1024 for math
EVAL_BATCH_SIZE=16
TEMPERATURE=0.0
SEED=42

# TokenSkip Settings
ADAPTER_PATH="./model/TokenSkip-Qwen2.5-7B-Instruct-GSM8K"
COMPRESSION_RATIO=0.5
for RATIO in 0.5 0.6 0.7 0.8 0.9 1.0; do
    python ./evaluation.py --output-dir "outputs/Qwen2.5-7B-Instruct/gsm8k/qwen_math_compressed/" \
        --model-path "./model/Qwen2.5-7B-Instruct" --tokenizer-path ${MODEL_PATH} \
        --model-size "7b" --model-type "qwen" --data-type "test"  \
        --max_num_examples 100000000000000 --max_new_tokens 512 \
        --eval_batch_size 32 --temperature 0.0 --seed 42 --benchmark "gsm8k" \
        --adapter-path "./model/lora_sft_qwen_math_Qwen_7B_lr_5e-5" \
        --compression_ratio ${RATIO} --use_adapter;
done