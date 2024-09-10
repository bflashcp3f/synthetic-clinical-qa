import sys
import os
import re
import openai
import json
import time
import argparse
import tiktoken

from collections import defaultdict, Counter
from pathlib import Path

from p2g.tasks import get_task
from p2g.methods.prompt import prompt_naive, prompt_icl


def main(args):

    # Set the OpenAI API key
    if args.api_source == 'openai':
        openai.api_key = os.getenv("OPENAI_API_KEY")
    elif args.api_source == 'azure':
        openai.api_type = "azure"
        openai.api_base = "https://inference.openai.azure.com/"
        openai.api_version = "2023-07-01-preview"
        openai.api_key = os.getenv("AZURE_OPENAI_KEY")
    elif args.api_source != 'open_source':
        raise ValueError(f"API source {args.api_source} not supported")
    
    METHOD_DICT = {
        'radqa': {
            'direct_instruction': {
                'llama3-8b': [
                    {'template_name': 'question_generation_plain', 'process_unit': 'document', 'question_num': 10, 'task_end_index': 64},
                    {'template_name': 'answer_generation_gpt4', 'process_unit': 'paragraph', 'prior_template_name': 'question_generation_plain', 'prior_process_unit': 'document', 'question_num': 10, 'task_end_index': 64},
                ],
                'gpt-4o-2024-05-13': [
                    {'template_name': 'question_generation_plain', 'process_unit': 'document', 'question_num': 10, 'task_end_index': 64},
                    {'template_name': 'answer_generation_gpt4', 'process_unit': 'paragraph', 'prior_template_name': 'question_generation_plain', 'prior_process_unit': 'document', 'question_num': 10, 'task_end_index': 64},
                ]
            },
            'direct_instruct_nonoverlap': {
                'llama3-8b': [
                    {'template_name': 'question_generation_nonoverlap', 'process_unit': 'document', 'question_num': 10, 'task_end_index': 64},
                    {'template_name': 'answer_generation_gpt4', 'process_unit': 'paragraph', 'prior_template_name': 'question_generation_nonoverlap', 'prior_process_unit': 'document', 'question_num': 10, 'task_end_index': 64},
                ],
                'gpt-4o-2024-05-13': [
                    {'template_name': 'question_generation_nonoverlap', 'process_unit': 'document', 'question_num': 10, 'task_end_index': 64},
                    {'template_name': 'answer_generation_gpt4', 'process_unit': 'paragraph', 'prior_template_name': 'question_generation_nonoverlap', 'prior_process_unit': 'document', 'question_num': 10, 'task_end_index': 64},
                ]
            },
            'direct_instruct_temp_anneal': {
                'llama3-8b': [
                    {'template_name': 'question_generation_plain', 'process_unit': 'document', 'question_num': 10, 'task_end_index': 64, 'temp_max': 1},
                    {'template_name': 'answer_generation_gpt4', 'process_unit': 'paragraph', 'prior_template_name': 'question_generation_plain', 'prior_process_unit': 'document', 'question_num': 10, 'task_end_index': 64, 'temp_max': 0},
                ],
                'gpt-4o-2024-05-13': [
                    {'template_name': 'question_generation_plain', 'process_unit': 'document', 'question_num': 10, 'task_end_index': 64, 'temp_max': 1},
                    {'template_name': 'answer_generation_gpt4', 'process_unit': 'paragraph', 'prior_template_name': 'question_generation_plain', 'prior_process_unit': 'document', 'question_num': 10, 'task_end_index': 64, 'temp_max': 0},
                ]
            },
            'explicit_prompt': {
                'llama3-8b': [
                    {'template_name': 'question_generation_explicit', 'process_unit': 'document', 'question_num': 10, 'task_end_index': 64},
                    {'template_name': 'answer_generation_gpt4', 'process_unit': 'paragraph', 'prior_template_name': 'question_generation_explicit', 'prior_process_unit': 'document', 'question_num': 10, 'task_end_index': 64},
                ],
                'gpt-4o-2024-05-13': [
                    {'template_name': 'question_generation_explicit', 'process_unit': 'document', 'question_num': 10, 'task_end_index': 64},
                    {'template_name': 'answer_generation_gpt4', 'process_unit': 'paragraph', 'prior_template_name': 'question_generation_explicit', 'prior_process_unit': 'document', 'question_num': 10, 'task_end_index': 64},
                ]
            },
            'summarization_direct_instruction': {
                'llama3-8b': [
                    {'template_name': 'summary_generation', 'process_unit': 'document', 'task_end_index': 64},
                    {'template_name': 'question_generation_summary', 'process_unit': 'document', 'prior_template_name': 'summary_generation', 'prior_process_unit': 'document', 'question_num': 10, 'task_end_index': 64},
                    {'template_name': 'answer_generation_gpt4', 'process_unit': 'paragraph', 'prior_template_name': 'question_generation_summary', 'prior_process_unit': 'document', 'question_num': 10, 'task_end_index': 64},
                ],
                'gpt-4o-2024-05-13': [
                    {'template_name': 'summary_generation', 'process_unit': 'document', 'task_end_index': 64},
                    {'template_name': 'question_generation_summary', 'process_unit': 'document', 'prior_template_name': 'summary_generation', 'prior_process_unit': 'document', 'question_num': 10, 'task_end_index': 64},
                    {'template_name': 'answer_generation_gpt4', 'process_unit': 'paragraph', 'prior_template_name': 'question_generation_summary', 'prior_process_unit': 'document', 'question_num': 10, 'task_end_index': 64},
                ]
            },
            'summarization_nonoverlap': {
                'llama3-8b': [
                    {'template_name': 'summary_generation', 'process_unit': 'document', 'task_end_index': 64},
                    {'template_name': 'question_generation_summary_nonoverlap', 'process_unit': 'document', 'prior_template_name': 'summary_generation', 'prior_process_unit': 'document', 'question_num': 10, 'task_end_index': 64},
                    {'template_name': 'answer_generation_gpt4', 'process_unit': 'paragraph', 'prior_template_name': 'question_generation_summary_nonoverlap', 'prior_process_unit': 'document', 'question_num': 10, 'task_end_index': 64},
                ],
                'gpt-4o-2024-05-13': [
                    {'template_name': 'summary_generation', 'process_unit': 'document', 'task_end_index': 64},
                    {'template_name': 'question_generation_summary_nonoverlap', 'process_unit': 'document', 'prior_template_name': 'summary_generation', 'prior_process_unit': 'document', 'question_num': 10, 'task_end_index': 64},
                    {'template_name': 'answer_generation_gpt4', 'process_unit': 'paragraph', 'prior_template_name': 'question_generation_summary_nonoverlap', 'prior_process_unit': 'document', 'question_num': 10, 'task_end_index': 64},
                ]
            },
        },
        'mimicqa': {
            'direct_instruction': {
                'llama3-8b': [
                    {'template_name': 'question_generation_plain', 'process_unit': 'paragraph', 'question_num': 20, 'task_end_index': 169},
                    {'template_name': 'answer_generation', 'process_unit': 'paragraph', 'prior_template_name': 'question_generation_plain', 'prior_process_unit': 'paragraph', 'question_num': 20, 'task_end_index': 169},
                ],
                'gpt-4o-2024-05-13': [
                    {'template_name': 'question_generation_plain', 'process_unit': 'paragraph', 'question_num': 20, 'task_end_index': 169},
                    {'template_name': 'answer_generation', 'process_unit': 'paragraph', 'prior_template_name': 'question_generation_plain', 'prior_process_unit': 'paragraph', 'question_num': 20, 'task_end_index': 169},
                ]
            },
            'direct_instruct_nonoverlap': {
                'llama3-8b': [
                    {'template_name': 'question_generation_nonoverlap', 'process_unit': 'paragraph', 'question_num': 20, 'task_end_index': 169},
                    {'template_name': 'answer_generation', 'process_unit': 'paragraph', 'prior_template_name': 'question_generation_nonoverlap', 'prior_process_unit': 'paragraph', 'question_num': 20, 'task_end_index': 169},
                ],
                'gpt-4o-2024-05-13': [
                    {'template_name': 'question_generation_nonoverlap', 'process_unit': 'paragraph', 'question_num': 20, 'task_end_index': 169},
                    {'template_name': 'answer_generation', 'process_unit': 'paragraph', 'prior_template_name': 'question_generation_nonoverlap', 'prior_process_unit': 'paragraph', 'question_num': 20, 'task_end_index': 169},
                ]
            },
            'direct_instruct_temp_anneal': {
                'llama3-8b': [
                    {'template_name': 'question_generation_plain', 'process_unit': 'paragraph', 'question_num': 20, 'task_end_index': 169, 'temp_max': 1},
                    {'template_name': 'answer_generation', 'process_unit': 'paragraph', 'prior_template_name': 'question_generation_plain', 'prior_process_unit': 'paragraph', 'question_num': 20, 'task_end_index': 169, 'temp_max': 0},
                ],
                'gpt-4o-2024-05-13': [
                    {'template_name': 'question_generation_plain', 'process_unit': 'paragraph', 'question_num': 20, 'task_end_index': 169, 'temp_max': 1},
                    {'template_name': 'answer_generation', 'process_unit': 'paragraph', 'prior_template_name': 'question_generation_plain', 'prior_process_unit': 'paragraph', 'question_num': 20, 'task_end_index': 169, 'temp_max': 0},
                ]
            },
            'explicit_prompt': {
                'llama3-8b': [
                    {'template_name': 'question_generation_explicit', 'process_unit': 'paragraph', 'question_num': 20, 'task_end_index': 169},
                    {'template_name': 'answer_generation', 'process_unit': 'paragraph', 'prior_template_name': 'question_generation_explicit', 'prior_process_unit': 'paragraph', 'question_num': 20, 'task_end_index': 169},
                ],
                'gpt-4o-2024-05-13': [
                    {'template_name': 'question_generation_explicit', 'process_unit': 'paragraph', 'question_num': 20, 'task_end_index': 169},
                    {'template_name': 'answer_generation', 'process_unit': 'paragraph', 'prior_template_name': 'question_generation_explicit', 'prior_process_unit': 'paragraph', 'question_num': 20, 'task_end_index': 169},
                ]
            },
            'summarization_direct_instruction': {
                'llama3-8b': [
                    {'template_name': 'summary_generation', 'process_unit': 'paragraph', 'task_end_index': 169},
                    {'template_name': 'question_generation_summary', 'process_unit': 'paragraph', 'prior_template_name': 'summary_generation', 'prior_process_unit': 'paragraph', 'question_num': 20, 'task_end_index': 169},
                    {'template_name': 'answer_generation', 'process_unit': 'paragraph', 'prior_template_name': 'question_generation_summary', 'prior_process_unit': 'paragraph', 'question_num': 20, 'task_end_index': 169},
                ],
                'gpt-4o-2024-05-13': [
                    {'template_name': 'summary_generation', 'process_unit': 'paragraph', 'task_end_index': 169},
                    {'template_name': 'question_generation_summary', 'process_unit': 'paragraph', 'prior_template_name': 'summary_generation', 'prior_process_unit': 'paragraph', 'question_num': 20, 'task_end_index': 169},
                    {'template_name': 'answer_generation', 'process_unit': 'paragraph', 'prior_template_name': 'question_generation_summary', 'prior_process_unit': 'paragraph', 'question_num': 20, 'task_end_index': 169},
                ]
            },
            'summarization_explicit': {
                'llama3-8b': [
                    {'template_name': 'summary_generation', 'process_unit': 'paragraph', 'task_end_index': 169},
                    {'template_name': 'question_generation_summary_explicit', 'process_unit': 'paragraph', 'prior_template_name': 'summary_generation', 'prior_process_unit': 'paragraph', 'question_num': 20, 'task_end_index': 169},
                    {'template_name': 'answer_generation', 'process_unit': 'paragraph', 'prior_template_name': 'question_generation_summary_explicit', 'prior_process_unit': 'paragraph', 'question_num': 20, 'task_end_index': 169},
                ],
                'gpt-4o-2024-05-13': [
                    {'template_name': 'summary_generation', 'process_unit': 'paragraph', 'task_end_index': 169},
                    {'template_name': 'question_generation_summary_explicit', 'process_unit': 'paragraph', 'prior_template_name': 'summary_generation', 'prior_process_unit': 'paragraph', 'question_num': 20, 'task_end_index': 169},
                    {'template_name': 'answer_generation', 'process_unit': 'paragraph', 'prior_template_name': 'question_generation_summary_explicit', 'prior_process_unit': 'paragraph', 'question_num': 20, 'task_end_index': 169},
                ]
            },
        },
    }
        
    assert args.method_name in METHOD_DICT[args.task]
    
    for exp_setting in METHOD_DICT[args.task][args.method_name][args.backend]:
        
        # Set the template
        args.template = exp_setting['template_name']
        args.process_unit = exp_setting['process_unit']
        args.prior_template = exp_setting.get('prior_template_name', None)
        args.prior_process_unit = exp_setting.get('prior_process_unit', None)
        args.temp_max = exp_setting.get('temp_max', args.temp_max)
        args.question_num = exp_setting.get('question_num', args.question_num)
        args.task_end_index = exp_setting.get('task_end_index', args.task_end_index)
        
        print(f"Arguments: {vars(args)}")
    
        # Get the task
        task = get_task(args)
        
        # Run the task
        if args.prompt_setting == 'naive':
            if not args.api_source == 'open_source':
                prompt_naive(args, task)
            else:
                prompt_naive(args, task, open_source=True)
        elif args.prompt_setting == 'icl':
            if not args.api_source == 'open_source':
                prompt_icl(args, task)
            else:
                prompt_icl(args, task, open_source=True)
        else:
            raise ValueError(f"Prompt setting {args.prompt_setting} not supported")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--backend', type=str, required=True)
    parser.add_argument('--api_source', type=str)
    
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--method_name', type=str, required=True)
    parser.add_argument('--template', type=str)
    parser.add_argument('--prior_template', type=str)
    parser.add_argument('--task_start_index', type=int, default=0)
    parser.add_argument('--task_end_index', type=int, default=0)
    parser.add_argument('--question_num', type=int, default=10)
    parser.add_argument('--paraphrase_num', type=int, default=1)
    parser.add_argument('--data_split', type=str, default='dev', required=True)
    parser.add_argument('--answerable_only', action='store_true')
    parser.add_argument('--unanswerable_only', action='store_true')
    parser.add_argument('--data_subset', type=str, default=None)
    parser.add_argument('--process_unit', type=str, default='paragraph', choices=['document', 'paragraph', 'qa_pair'])
    parser.add_argument('--prior_process_unit', type=str, choices=['document', 'paragraph', 'qa_pair'])

    parser.add_argument('--temp_min', type=float, default=0)
    parser.add_argument('--temp_max', type=float, default=0)
    parser.add_argument('--max_tokens', type=int, default=256)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--stop', type=list, default=[])

    parser.add_argument('--sleep_time', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--prompt_setting', type=str, choices=['naive', 'icl'], required=True)

    parser.add_argument('--icl_style', type=str, default='random', choices=['random', 'retrieval'])
    parser.add_argument('--retrieval_model', type=str)
    parser.add_argument('--embed_text', type=str)
    parser.add_argument('--icl_random_seed', type=int, default=42)
    parser.add_argument('--icl_start_index', type=int, default=-1)
    parser.add_argument('--icl_end_index', type=int, default=-1)
    parser.add_argument('--icl_num', type=int, default=1)
    
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--async_prompt', action='store_true')
    
    args = parser.parse_args()
    
    if args.backend in ['llama3-8b']:
        args.api_source = 'open_source'
    else:
        args.api_source = 'azure'
    
    main(args)
