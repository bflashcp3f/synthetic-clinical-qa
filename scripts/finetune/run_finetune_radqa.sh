#!/bin/bash

# conda env: qa
# command: bash scripts/finetune/run_finetune_radqa.sh llama3-8b direct_instruction 0_64 generate_10_select_5 42
# command: bash scripts/finetune/run_finetune_radqa.sh gpt-4o-2024-05-13 direct_instruction 0_64 generate_10_select_5 42

MODEL_NAME=$1
METHOD_NAME=$2
SUBSET=$3
EXP_NAME=$4
RANDOM_SEED=$5

TRAIN_FILE="data/output/radqa/${METHOD_NAME}/train/${SUBSET}/${MODEL_NAME}/data_${EXP_NAME}.csv"
OUTPUT_DIR="output/checkpoints/radqa_${MODEL_NAME}_${METHOD_NAME}_${SUBSET}_${EXP_NAME}_ft_seed_${RANDOM_SEED}/"

# Check if an argument is provided for CUDA_VISIBLE_DEVICES
if [ -z "$6" ]; then
  # If no argument is provided, set CUDA_VISIBLE_DEVICES to zero
  export CUDA_VISIBLE_DEVICES=0
else
  # If an argument is provided, use it for CUDA_VISIBLE_DEVICES
  export CUDA_VISIBLE_DEVICES=$6
fi

python code/finetune_radqa_encoder.py \
    --model_name_or_path models/BioClinRoBERTa/RoBERTa-large-PM-M3-Voc \
    --train_file $TRAIN_FILE \
    --validation_file "data/modified/radqa/dev_processed.csv" \
    --test_file "data/modified/radqa/test_processed.csv" \
    --version_2_with_negative \
    --do_train \
    --do_eval \
    --do_predict \
    --seed $RANDOM_SEED \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --num_train_epochs 40 \
    --save_total_limit 1 \
    --metric_for_best_model "eval_f1" \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --load_best_model_at_end \
    --overwrite_output_dir \
    --max_seq_length 512 \
    --doc_stride 128 \
    --report_to "none" \
    --output_dir $OUTPUT_DIR


