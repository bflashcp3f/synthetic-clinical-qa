#!/bin/bash

# conda env: qa
# command: bash scripts/preprocess/process_llm_output_radqa.sh llama3-8b direct_instruction answer_generation_gpt4 20 10
# command: bash scripts/preprocess/process_llm_output_radqa.sh gpt-4o-2024-05-13 direct_instruction answer_generation_gpt4 20 10

# command: python code/process_llm_output.py --task radqa --data_split train --model_name gpt-4o-2024-05-13 --method_name summarization_nonoverlap --template_name answer_generation_gpt4 --question_num 20 --qa_select_num 20

DATA_SPLIT=train
MODEL_NAME=$1
METHOD_NAME=$2
TEMPLATE_NAME=$3
QUESTION_NUM=$4
QA_SELECT_NUM=$5
SUBSET="0_64"

python code/process_llm_output.py \
    --task radqa \
    --data_split $DATA_SPLIT \
    --model_name $MODEL_NAME \
    --method_name $METHOD_NAME \
    --template_name $TEMPLATE_NAME \
    --question_num $QUESTION_NUM \
    --qa_select_num $QA_SELECT_NUM \
    --subset $SUBSET