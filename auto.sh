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
AUTO_RATIO=0.2


python ./make_auto_dataset.py --input-dir "outputs/Qwen2.5-7B-Instruct/gsm8k/auto1/${AUTO_RATIO}/-1/7b/TokenSkip" --output-dir "data/auto_dataset/${AUTO_RATIO}/-1" --balance-weight -1

python ./merge_auto_and_train.py --first-dataset "data/auto_dataset/${AUTO_RATIO}/-1/auto_dataset.json" --second-dataset data/mydataset_compressed_gsm8k_llmlingua2_qwen_7B.json --output data/merged_dataset.json
    
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/examples/train_lora/myllama3_lora_sft_compressed_gsm8k_llmlingua2_qwen_7B_auto2.yaml

cp original_datasets/gsm8k/test.jsonl datasets/gsm8k
    
python ./evaluation.py --output-dir "outputs/Qwen2.5-7B-Instruct/gsm8k/auto2/${AUTO_RATIO}/-1" \
        --model-path "./model/Qwen2.5-7B-Instruct" --tokenizer-path ${MODEL_PATH} \
        --model-size "7b" --model-type "qwen" --data-type "test"  \
        --max_num_examples 100000000000000 --max_new_tokens 512 \
        --eval_batch_size 32 --temperature 0.0 --seed 42 --benchmark "gsm8k" \
        --adapter-path "./model/lora_sft_llmlingua2_Qwen_7B_lr_5e-5_auto2" \
        --compression_ratio 0.0 --use_adapter --auto_gamma

for BALANCE_WEIGHT in -2 0 1 2; do
# Auto Settings

    python process_dataset_inone.py --ratio ${AUTO_RATIO} --shuffle --seed 999
    
    #CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/examples/train_lora/myllama3_lora_sft_compressed_gsm8k_llmlingua2_qwen_7B_auto1.yaml
    
    # for COMPRESSION_RATIO in 0.5 0.6 0.7 0.8 0.9 1.0; do
    #     python ./evaluation.py --output-dir "outputs/Qwen2.5-7B-Instruct/gsm8k/auto1/${AUTO_RATIO}/${BALANCE_WEIGHT}" \
    #         --model-path "./model/Qwen2.5-7B-Instruct" --tokenizer-path ${MODEL_PATH} \
    #         --model-size "7b" --model-type "qwen" --data-type "test"  \
    #         --max_num_examples 100000000000000 --max_new_tokens 512 \
    #         --eval_batch_size 32 --temperature 0.0 --seed 42 --benchmark "gsm8k" \
    #         --adapter-path "./model/lora_sft_llmlingua2_Qwen_7B_lr_5e-5_auto1" \
    #         --compression_ratio ${COMPRESSION_RATIO} --use_adapter;
    # done

    python ./make_auto_dataset.py --input-dir "outputs/Qwen2.5-7B-Instruct/gsm8k/auto1/${AUTO_RATIO}/-1/7b/TokenSkip" --output-dir "data/auto_dataset/${AUTO_RATIO}/${BALANCE_WEIGHT}" --balance-weight ${BALANCE_WEIGHT}

    python ./merge_auto_and_train.py --first-dataset "data/auto_dataset/${AUTO_RATIO}/${BALANCE_WEIGHT}/auto_dataset.json" --second-dataset data/mydataset_compressed_gsm8k_llmlingua2_qwen_7B.json --output data/merged_dataset.json
    
    CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/examples/train_lora/myllama3_lora_sft_compressed_gsm8k_llmlingua2_qwen_7B_auto2.yaml

    cp original_datasets/gsm8k/test.jsonl datasets/gsm8k
    
    python ./evaluation.py --output-dir "outputs/Qwen2.5-7B-Instruct/gsm8k/auto2/${AUTO_RATIO}/${BALANCE_WEIGHT}" \
        --model-path "./model/Qwen2.5-7B-Instruct" --tokenizer-path ${MODEL_PATH} \
        --model-size "7b" --model-type "qwen" --data-type "test"  \
        --max_num_examples 100000000000000 --max_new_tokens 512 \
        --eval_batch_size 32 --temperature 0.0 --seed 42 --benchmark "gsm8k" \
        --adapter-path "./model/lora_sft_llmlingua2_Qwen_7B_lr_5e-5_auto2" \
        --compression_ratio 0.0 --use_adapter --auto_gamma
done