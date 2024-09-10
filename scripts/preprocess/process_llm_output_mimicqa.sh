#!/bin/bash

# conda env: qa
# command: bash scripts/preprocess/process_llm_output_mimicqa.sh llama3-8b direct_instruction answer_generation 20 5
# command: bash scripts/preprocess/process_llm_output_mimicqa.sh gpt-4o-2024-05-13 direct_instruction answer_generation 20 5

# command: python code/process_llm_output.py --task mimicqa --output_path_json data/modified/mimicqa/test_processed.json

DATA_SPLIT=train
MODEL_NAME=$1
METHOD_NAME=$2
TEMPLATE_NAME=$3
QUESTION_NUM=$4
QA_SELECT_NUM=$5
SUBSET="0_169"

python code/process_llm_output.py \
    --task mimicqa \
    --data_split $DATA_SPLIT \
    --model_name $MODEL_NAME \
    --method_name $METHOD_NAME \
    --template_name $TEMPLATE_NAME \
    --question_num $QUESTION_NUM \
    --qa_select_num $QA_SELECT_NUM \
    --subset $SUBSET