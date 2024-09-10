#!/bin/bash

# conda env: qa
# command: bash scripts/generate/run_generate.sh radqa direct_instruction llama3-8b
# command: bash scripts/generate/run_generate.sh radqa direct_instruction gpt-4o-2024-05-13

# command: bash scripts/generate/run_generate.sh mimicqa direct_instruction llama3-8b
# command: bash scripts/generate/run_generate.sh mimicqa direct_instruction gpt-4o-2024-05-13

BATCH_SIZE=5
SLEEP_TIME=5
MAX_TOKENS=1024

TASK=$1
METHOD_NAME=$2
BACKEND=$3

python run.py \
    --backend $BACKEND \
    --task $TASK \
    --prompt_setting naive \
    --method_name $METHOD_NAME \
    --data_split train \
    --max_tokens $MAX_TOKENS \
    --batch_size $BATCH_SIZE \
    --sleep_time $SLEEP_TIME \
    --verbose \
    --async_prompt
