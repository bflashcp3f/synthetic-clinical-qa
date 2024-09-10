import json
import string
import ast
import re
import argparse
import random

from pathlib import Path
from collections import Counter

from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords

from pandas import DataFrame
import pandas as pd

from scipy.stats import skew
from evaluate import load

def print_result(result):
    # print(result)
    if 'NoAns_total' in result:
        num_total, num_ans, num_unans = result['total'], result['HasAns_total'], result['NoAns_total']
        print(f"em ({num_total}): {result['exact']:.1f}")
        print(f"f1 ({num_total}): {result['f1']:.1f}")
        if 'pm' in result:
            print(f"pm ({num_total}): {result['pm']:.1f}")
        print(f"f1 answerable ({num_ans}): {result['HasAns_f1']:.1f}")
        print(f"f1 unanswerable ({num_unans}): {result['NoAns_f1']:.1f}")
        print()
    else:
        num_total = result['total']
        print(f"em ({num_total}): {result['exact']:.1f}")
        print(f"f1 ({num_total}): {result['f1']:.1f}")
        print(f"pm ({num_total}): {result['pm']:.1f}")
    
def generate_unigrams(text):
    """
    Function to generate unigrams from a given text.
    """
    # Normalize the text
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the text
    tokens = text.split()

    # Unigrams are essentially the tokens themselves
    unigrams = tokens
    return unigrams

def find_all_occurrences(text, substring):
    indices = []
    start = 0
    while True:
        index = text.find(substring, start)
        if index == -1:
            break
        indices.append((index, index + len(substring)))
        start = index + 1  # Move to the next position after the found substring
    return indices

def has_overlapping_ranges(list1, list2):
    def ranges_overlap(range1, range2):
        return range1[0] < range2[1] and range2[0] < range1[1]
    
    for range1 in list1:
        for range2 in list2:
            if ranges_overlap(range1, range2):
                return True
    return False

def calculate_percent_overlap_ans(pred_labels, gold_labels, contexts):
    
    num_overlap_ans = 0
    for pred_label, gold_label, context in zip(sorted(pred_labels, key=lambda x: x['id']), sorted(gold_labels, key=lambda x: x['id']), sorted(contexts, key=lambda x: x['id'])):
        assert pred_label['id'] == gold_label['id']
        assert pred_label['id'] == context['id']
        
        pred_ans = pred_label['prediction_text']
        pred_ans_tokens = [token for token in generate_unigrams(pred_ans) if token not in stopwords.words('english')]
        # print(gold_label)
        if gold_label['answers']:
            if type(gold_label['answers']) == list:
                gold_ans = gold_label['answers'][0]['text']
            elif type(gold_label['answers']) == dict:
                gold_ans = gold_label['answers']['text'][0]
            else:
                raise ValueError("Gold answer format not recognized")
        else:
            gold_ans = ''
        gold_ans_tokens = [token for token in generate_unigrams(gold_ans) if token not in stopwords.words('english')]
        
        if pred_ans == gold_ans:
            num_overlap_ans += 1
            continue
        elif (pred_ans == '' and gold_ans != '') or (pred_ans != '' and gold_ans == ''):
            continue
        else:
            gold_ans_context_index_list = find_all_occurrences(context['context'], gold_ans)
            assert gold_ans_context_index_list
            
            pred_ans_context_index_list = find_all_occurrences(context['context'], pred_ans)
            assert pred_ans_context_index_list
            
            # if has_overlapping_ranges(gold_ans_context_index_list, pred_ans_context_index_list):
            #     num_overlap_ans += 1
        
            overlap_tokens = set(pred_ans_tokens).intersection(set(gold_ans_tokens))
            if overlap_tokens:
                num_overlap_ans += 1
            
    # print(f"{num_overlap_ans/len(pred_labels)*100:.1f}% of the answers overlap with the gold answers")
    # print(f"pm ({len(pred_labels)}): {num_overlap_ans/len(pred_labels)*100:.1f}")
    return num_overlap_ans/len(pred_labels)*100

def main(args):
    
    squad_v2_metric = load("squad_v2")
    
    qa_gold_path = Path(f'data/modified/{args.task}/{args.data_split}_processed.json')
    with open(qa_gold_path) as f:
        qa_gold = json.load(f)
    print(qa_gold_path)
    # breakpoint()
    
    if args.qa_pred_dir is not None:
        qa_pred_path = Path(args.qa_pred_dir) / 'predict_predictions.json'
    else:
        if args.subset is not None:
            if args.ft_seed is not None:
                qa_pred_path = Path(f'output/{args.task}_{args.model_name}_{args.method_name}_{args.subset}_generate_{args.question_num}_select_{args.qa_select_num}_ft_seed_{args.ft_seed}') / 'predict_predictions.json'
            else:
                qa_pred_path = Path(f'output/{args.task}_{args.model_name}_{args.method_name}_{args.subset}_generate_{args.question_num}_select_{args.qa_select_num}') / 'predict_predictions.json'
        else:
            if args.ft_seed is not None:
                qa_pred_path = Path(f'output/{args.task}_{args.model_name}_{args.method_name}_generate_{args.question_num}_select_{args.qa_select_num}_ft_seed_{args.ft_seed}') / 'predict_predictions.json'
            else:
                qa_pred_path = Path(f'output/{args.task}_{args.model_name}_{args.method_name}_generate_{args.question_num}_select_{args.qa_select_num}') / 'predict_predictions.json'
    print(qa_pred_path)
        
    with open(qa_pred_path) as f:
        qa_pred = json.load(f)
        
    gold_labels = []
    pred_labels = []
    contexts = []
    gold_overlap_dict = {}
    
    for data_item in qa_gold['data']:
        
        for paragraph in data_item['paragraphs']:
            
            context = paragraph['context']
            context_tokens = [token for token in generate_unigrams(context) if token not in stopwords.words('english')]
            
            for qa in paragraph['qas']:
                
                id = qa['id']
                
                question = qa['question']
                question_tokens = [token for token in generate_unigrams(question) if token not in stopwords.words('english')]
                
                overlap_tokens = set(question_tokens).intersection(set(context_tokens))
        
                if overlap_tokens:
                    gold_overlap_dict[id] = True
                else:
                    gold_overlap_dict[id] = False
                
                answers = qa['answers']
                
                gold_labels.append({'id': id, 'answers': [{'text': a['text'], 'answer_start': a['answer_start']} for a in answers]})
                assert all([a['text'] in context for a in answers])
                
                answer_pred_text = qa_pred[id]
                assert answer_pred_text in context
                
                if not answer_pred_text:
                    pred_labels.append({'id': id, 'prediction_text': '', 'no_answer_probability': 1.0})
                else:
                    pred_labels.append({'id': id, 'prediction_text': answer_pred_text, 'no_answer_probability': 0.0})
                    
                contexts.append({'id': id, 'context': context})
        
    print(f"Overall performance on {len(gold_labels)} questions")
    assert len(gold_labels) == len(pred_labels) == len(contexts)
    pm = calculate_percent_overlap_ans(pred_labels, gold_labels, contexts)
    results = squad_v2_metric.compute(predictions=pred_labels, references=gold_labels)
    results['pm'] = pm
    # print(results)
    print_result(results)
    
    # if args.task == 'radqa':
    
    #     pred_labels_overlap = [item for item in pred_labels if gold_overlap_dict[item['id']]]
    #     gold_labels_overlap = [item for item in gold_labels if gold_overlap_dict[item['id']]]
        
    #     print(f"Performance on {len(pred_labels_overlap)} overlapping questions")
    #     results_overlap = squad_v2_metric.compute(predictions=pred_labels_overlap, references=gold_labels_overlap)
    #     print_result(results_overlap)

    #     pred_labels_nonoverlap = [item for item in pred_labels if not gold_overlap_dict[item['id']]]
    #     gold_labels_nonoverlap = [item for item in gold_labels if not gold_overlap_dict[item['id']]]
        
    #     print(f"Performance on {len(pred_labels_nonoverlap)} non-overlapping questions")
    #     results_nonoverlap = squad_v2_metric.compute(predictions=pred_labels_nonoverlap, references=gold_labels_nonoverlap)
    #     print_result(results_nonoverlap)
    # elif args.task == 'mimicqa':
        
    #     id_mapping_path = Path(f'data/modified/mimicqa/id_mapping.json')
    #     with open(id_mapping_path) as f:
    #         id_mapping = json.load(f)
            
    #     # pred_labels_hg = [item for item in pred_labels if int(id_mapping[item['id']]) >999]
    #     # gold_labels_hg = [item for item in gold_labels if int(id_mapping[item['id']]) >999]
        
    #     # print(f"Performance on {len(pred_labels_hg)} human-generated questions")
    #     # calculate_percent_overlap_ans(pred_labels_hg, gold_labels_hg)
    #     # results = squad_v2_metric.compute(predictions=pred_labels_hg, references=gold_labels_hg)
    #     # print_result(results)
        
    #     # pred_labels_hv = [item for item in pred_labels if int(id_mapping[item['id']]) <=999]
    #     # gold_labels_hv = [item for item in gold_labels if int(id_mapping[item['id']]) <=999]
        
    #     # print(f"Performance on {len(pred_labels_hv)} human-validated questions")
    #     # calculate_percent_overlap_ans(pred_labels_hv, gold_labels_hv)
    #     # results = squad_v2_metric.compute(predictions=pred_labels_hv, references=gold_labels_hv)
    #     # print_result(results)
        
    #     num_pm_bigger, num_f1_bigger, num_same = 0, 0, 0
    #     for pred_label, gold_label, context in zip(pred_labels, gold_labels, contexts):
            
    #         f1 = squad_v2_metric.compute(predictions=[pred_label], references=[gold_label])['f1']
    #         pm = calculate_percent_overlap_ans([pred_label], [gold_label], [context])
    #         # print(f"pm: {pm:.1f}, f1: {f1:.1f}")
    #         pred_text = pred_label['prediction_text']
    #         gold_text = gold_label['answers'][0]['text']
    #         # print(f"pred_label: {pred_text}")
    #         # print(f"gold_label: {gold_text}")
    #         # break
            
    #         if pm > f1:
    #             num_pm_bigger += 1
    #         elif f1 > pm:
    #             num_f1_bigger += 1
    #         else:
    #             num_same += 1
            
    #     assert len(gold_labels) == len(pred_labels) == num_pm_bigger + num_f1_bigger + num_same
    #     print(f"num_pm_bigger: {num_pm_bigger}, num_f1_bigger: {num_f1_bigger}, num_same: {num_same}")
                
        



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--data_split', type=str, default='test')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--method_name', type=str)
    parser.add_argument('--subset', type=str)
    parser.add_argument('--question_num', type=int, default=10)
    parser.add_argument('--qa_select_num', type=int, default=10)
    parser.add_argument('--ft_seed', type=int)
    parser.add_argument('--qa_pred_dir', type=str)
    
    args = parser.parse_args()
    main(args)