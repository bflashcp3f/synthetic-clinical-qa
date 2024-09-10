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

def get_qa_distribution(data):
    qa_distribution = {}
    for item in data['data']:
        for paragraph in item['paragraphs']:
            document_id = paragraph['document_id']
            num_ans = len([qa for qa in paragraph['qas'] if not qa['is_impossible']])
            num_unans = len([qa for qa in paragraph['qas'] if qa['is_impossible']])
            qa_distribution[document_id] = {'num_ans': num_ans, 'num_unans': num_unans, 'total': num_ans + num_unans}
                
    return qa_distribution


def process_answers(answer_span):
    answer_span = answer_span.strip()
    answer_span = answer_span.strip('.').strip()
    
    if 'there' in answer_span.lower():
        answer_span = answer_span[answer_span.lower().index('there'):].strip()
    
    if answer_span.lower().startswith('there'):
        # answer_span = answer_span.strip('There is').strip('There are').strip('There was').strip('There were').strip('there is').strip('there are').strip('there was').strip('there were').strip()
        
        # Replace 'There is' on the left with an empty string
        for starter_phrase in ['There is', 'There are', 'There was', 'There were', 'there is', 'there are', 'there was', 'there were']:
            if answer_span.startswith(starter_phrase):
                answer_span = answer_span[len(starter_phrase):].strip()
                break
    
    return answer_span


def remove_prefix(text):
    """
    Remove numerical prefixes at the start of a single-line string.

    Args:
        text (str): The input string with a numerical prefix.

    Returns:
        str: The string with the numerical prefix removed.
    """
    # Regular expression pattern to match the prefix at the start of the string
    pattern = r'^\d+\.\s+'
    
    # Use re.sub to remove the prefix at the start of the string
    cleaned_text = re.sub(pattern, '', text)
    
    return cleaned_text.strip()


def process_generated_qa_radqa(qa_pred_path, qa_select_num):
    
    print(f"Reading from {qa_pred_path}")
    with open(qa_pred_path) as f:
        data_generated = [json.loads(line) for line in f]
    
    num_unanswerable_all, num_valid_all, num_invalid_all, num_all, num_select_all, num_unanswerable_selcted_all = 0, 0, 0, 0, 0, 0

    data_generated_processed = []
    for data_item in data_generated:
        
        id = data_item['document_id']
        title = id.split('_')[0]
        context_str = data_item['context']
        qas_generated_str = data_item['qas_generated']
        
        qas_generated = qas_generated_str.split('\n\n')
        qas_generated_processed = []
        
        num_unanswerable, num_valid, num_invalid = 0, 0, 0
        
        for qa_idx, qa_item in enumerate(qas_generated):
            
            if len(qa_item.split('\n')) != 2:
                num_invalid += 1
                continue
            
            q_str, a_str = qa_item.split('\n')
            # print([q_str, a_str], '\n')
            
            if not q_str.startswith('Q:'):
                print(f"q_str does not start with 'Q:' for doc {id}, qa_idx {qa_idx}")
                print([q_str, a_str], '\n')
                num_invalid += 1
                continue
            # assert q_str.startswith('Q:')
            
            if not a_str.startswith('A:'):
                print(f"a_str does not start with 'A:' for doc {id}, qa_idx {qa_idx}")
                print([q_str, a_str], '\n')
                num_invalid += 1
                continue
            # assert a_str.startswith('A:')
            
            q_str = q_str[2:].strip()
            a_str = a_str[2:].strip().strip('"').strip()
            
            q_str = remove_prefix(q_str)
            
            if not q_str or not a_str:
                num_invalid += 1
                continue
            
            q_id = f"{id}_Q{qa_idx}"
            a_id = f"{id}_Q{qa_idx}_A{qa_idx}"
            
            # print(f"Q: {q_str}\nA: {a_str}\n")
            
            if a_str.startswith("Unanswerable"):
                num_unanswerable += 1
                
                answers = []
                qa_item_generated = {
                    'id': q_id,
                    'question': q_str,
                    'answers': answers,
                    'is_impossible': True
                }
                qas_generated_processed.append(qa_item_generated)
            else:
                if a_str in context_str:
                    a_str = process_answers(a_str)
                    
                    if not a_str:
                        num_invalid += 1
                        continue
                    
                    assert a_str in context_str
                    
                    num_valid += 1
                
                    answers = [{
                        'answer_id': a_id,
                        'text': a_str,
                        'answer_start': context_str.index(a_str)
                    }]
                    assert a_str == context_str[answers[0]['answer_start']:answers[0]['answer_start']+len(a_str)]
                    qa_item_generated = {
                        'id': q_id,
                        'question': q_str,
                        'answers': answers,
                        'is_impossible': False
                    }
                    qas_generated_processed.append(qa_item_generated)
                
                else:
                    num_invalid += 1
                   
        assert len(qas_generated_processed) == num_unanswerable+num_valid
         
        data_item_processed = {
            'title': title,
            'paragraphs': [
                {
                    'document_id': id,
                    'context': context_str,
                    'qas': qas_generated_processed[:qa_select_num]
                }
            ]
        }
        
        if len(data_item_processed['paragraphs'][0]['qas']) < qa_select_num:
            print(len(qas_generated), len(data_item_processed['paragraphs'][0]['qas']), qa_select_num)
            print(id, [qas_generated_str])
        
        data_generated_processed.append(data_item_processed)
        
        num_unanswerable_all += num_unanswerable
        num_valid_all += num_valid
        num_invalid_all += num_invalid
        num_all += len(qas_generated)
        num_select_all += len(data_item_processed['paragraphs'][0]['qas'])
        
        num_unanswerable_selcted = sum([qa['is_impossible'] for qa in data_item_processed['paragraphs'][0]['qas']])
        num_unanswerable_selcted_all += num_unanswerable_selcted
        
    print(f"For {len(data_generated)} documents, out of {num_all} generated QA pairs, {num_unanswerable_all} unanswerable, {num_valid_all} answerable, {num_invalid_all} invalid, {num_select_all} selected, {num_unanswerable_selcted_all} selected unanswerable ({num_unanswerable_selcted_all/num_select_all*100:.1f}%)")

    data_generated_processed = {
        'data': data_generated_processed
    }
    
    return data_generated_processed


def process_generated_qa_mimicqa(qa_pred_path, qa_select_num):
    
    print(f"Reading from {qa_pred_path}")
    with open(qa_pred_path) as f:
        data_generated = [json.loads(line) for line in f]
    
    num_unanswerable_all, num_valid_all, num_invalid_all, num_all, num_select_all = 0, 0, 0, 0, 0

    data_generated_processed = []
    for data_item in data_generated:
        
        id = data_item['document_id']
        title = id.split('_')[0]
        context_str = data_item['context']
        qas_generated_str = data_item['qas_generated']
        
        qas_generated = qas_generated_str.split('\n\n')
        qas_generated_processed = []
        
        num_unanswerable, num_valid, num_invalid = 0, 0, 0
        
        for qa_idx, qa_item in enumerate(qas_generated):
            
            if len(qa_item.split('\n')) != 2:
                num_invalid += 1
                continue
            
            q_str, a_str = qa_item.split('\n')
            # print([q_str, a_str], '\n')
            
            if not q_str.startswith('Q:'):
                print(f"q_str does not start with 'Q:' for doc {id}, qa_idx {qa_idx}")
                print([q_str, a_str], '\n')
                num_invalid += 1
                continue
            # assert q_str.startswith('Q:')
            
            if not a_str.startswith('A:'):
                print(f"a_str does not start with 'A:' for doc {id}, qa_idx {qa_idx}")
                print([q_str, a_str], '\n')
                num_invalid += 1
                continue
            # assert a_str.startswith('A:')
            
            q_str = q_str[2:].strip().lower().strip('?')
            a_str = a_str[2:].strip().strip('"').strip().lower()
            
            q_str = remove_prefix(q_str)
            
            if not q_str or not a_str:
                num_invalid += 1
                continue
            
            q_id = f"{id}_Q{qa_idx}"
            a_id = f"{id}_Q{qa_idx}_A{qa_idx}"
            
            # print(f"Q: {q_str}\nA: {a_str}\n")
            
            if a_str and a_str in context_str:
                # a_str = process_answers(a_str)
                
                assert a_str in context_str
                
                num_valid += 1
            
                answers = [{
                    'answer_id': a_id,
                    'text': a_str,
                    'answer_start': context_str.index(a_str)
                }]
                assert a_str == context_str[answers[0]['answer_start']:answers[0]['answer_start']+len(a_str)]
                qa_item_generated = {
                    'id': q_id,
                    'question': q_str,
                    'answers': answers,
                }
                qas_generated_processed.append(qa_item_generated)
            
            else:
                num_invalid += 1
                    
        assert len(qas_generated_processed) == num_unanswerable+num_valid
        data_item_processed = {
            'title': title,
            'paragraphs': [
                {
                    'document_id': id,
                    'context': context_str,
                    'qas': qas_generated_processed[:qa_select_num]
                }
            ]
        }
        # assert len(data_item_processed['paragraphs'][0]['qas']) == qa_select_num
        
        if len(data_item_processed['paragraphs'][0]['qas']) < qa_select_num:
            print(id, len(qas_generated), len(data_item_processed['paragraphs'][0]['qas']), qa_select_num)
        
        data_generated_processed.append(data_item_processed)
        
        num_unanswerable_all += num_unanswerable
        num_valid_all += num_valid
        num_invalid_all += num_invalid
        num_all += len(qas_generated)
        num_select_all += len(data_item_processed['paragraphs'][0]['qas'])
        
    print(f"For {len(data_generated)} documents, out of {num_all} generated QA pairs, {num_unanswerable_all} unanswerable, {num_valid_all} answerable, {num_invalid_all} invalid, {num_select_all} selected")

    data_generated_processed = {
        'data': data_generated_processed
    }
    
    return data_generated_processed


def extract_questions(response):
    
    # Regular expression to find a JSON-like object
    pattern = r'^\d+\.\s.*\?'
    
    # Find all matches with multiline flag enabled
    questions = re.findall(pattern, response, re.M)
    if len(questions) < 1:
        print(f"Expected at least 1 match, got {len(questions)} from response below:\n{response}")
    
    return '\n'.join(questions)


def process_paraphrased_questions(qa_pred_path):
    
    random.seed(args.random_seed)
    print(f"Reading from {qa_pred_path}")
    with open(qa_pred_path) as f:
        data_generated = [json.loads(line) for line in f]
    
    num_unanswerable_all, num_valid_all, num_invalid_all, num_all = 0, 0, 0, 0

    data_generated_processed = []
    for data_item in data_generated:
        
        id = data_item['id']
        title = id.split('_')[0]
        q_id = id
        a_id = id + '_A0'
        context_str = data_item['context']
        question = data_item['question']
        answer = data_item['answer']
        
        num_unanswerable, num_valid, num_invalid = 0, 0, 0
        
        if data_item['output'] is not None:
            question_paraphrased = extract_questions(data_item['output']).split('\n')
        else:
            question_paraphrased = []
            num_invalid += 1
            
        question_paraphrased = [remove_prefix(q.strip()) for q in question_paraphrased if q.strip()]
        
        qas_generated_processed = []
        
        # Sample a question from the paraphrased questions
        # q_str = random.choice([question] + question_paraphrased)
        # q_index = ([question] + question_paraphrased).index(q_str)
        
        if question_paraphrased:
            q_str = random.choice(question_paraphrased)
            q_index = question_paraphrased.index(q_str)
        else:
            q_str = question
            q_index = 0
            
        # print(f"Selected question {q_index}: {q_str}")
        a_str = answer[0]['text']
        
        if a_str.startswith("Unanswerable"):
            num_unanswerable += 1
            
            answers = []
            qa_item_generated = {
                'id': q_id,
                'question': q_str,
                'answers': answers,
                'is_impossible': True
            }
            qas_generated_processed.append(qa_item_generated)
        else:
            if a_str in context_str:
                # a_str = process_answers(a_str)
                
                if not a_str:
                    num_invalid += 1
                    continue
                
                assert a_str in context_str
                
                num_valid += 1
            
                answers = [{
                    'answer_id': a_id,
                    'text': a_str,
                    'answer_start': context_str.index(a_str)
                }]
                assert a_str == context_str[answers[0]['answer_start']:answers[0]['answer_start']+len(a_str)]
                qa_item_generated = {
                    'id': q_id,
                    'question': q_str,
                    'answers': answers,
                    'is_impossible': False
                }
                qas_generated_processed.append(qa_item_generated)
            
            else:
                num_invalid += 1
                assert ValueError(f"Answer {a_str} not found in context {context_str}")
                    
        data_item_processed = {
            'title': title,
            'paragraphs': [
                {
                    'document_id': id,
                    'context': context_str,
                    'qas': qas_generated_processed
                }
            ]
        }
        assert len(qas_generated_processed) == num_unanswerable+num_valid
        
        data_generated_processed.append(data_item_processed)
        
        num_unanswerable_all += num_unanswerable
        num_valid_all += num_valid
        num_invalid_all += num_invalid
        num_all += 1
        
    print(f"For {len(data_generated)} documents, out of {num_all} generated QA pairs, {num_unanswerable_all} unanswerable, {num_valid_all} answerable, {num_invalid_all} invalid")

    data_generated_processed = {
        'data': data_generated_processed
    }
    
    return data_generated_processed


def convert_squad_to_csv_fmt(jf: dict):
    """Convert file to csv format. """
    new_data = {'id': [], 'title': [], 'context': [], 'question': [], 'answers': []}    

    data = jf['data']
    
    # num_answerable = len([qa for note in data for para in note['paragraphs'] for qa in para['qas'] if not qa['is_impossible']])
    # num_unanswerable = len([qa for note in data for para in note['paragraphs'] for qa in para['qas'] if qa['is_impossible']])
    # print(f"Out of {num_answerable+num_unanswerable} QA pairs, {num_answerable} answerable, {num_unanswerable} unanswerable")
    
    for note in data:
        title = note['title']
        for para in note['paragraphs']:
            context = para['context']

            for qa in para['qas']:
                
                new_data['id'].append(qa['id'])
                new_data['title'].append(title)
                new_data['context'].append(context)
                
                if qa['question'].startswith('**'):
                    question_str = qa['question'].replace('**', '')
                else:
                    question_str = qa['question']
                    
                new_data['question'].append(question_str)
                
                if 'is_impossible' in qa and qa['is_impossible']:
                    new_data['answers'].append({'text': [], 'answer_start': []})
                    continue 
    
                to_add = {'text': [], 'answer_start': []}
                for a in qa['answers']:
                    to_add['text'].append(a['text'])
                    to_add['answer_start'].append(a['answer_start'])
    
                new_data['answers'].append(to_add)
                
    return new_data
    

def main(args):
    
    if args.output_path_json is None:
        output_dir = Path(f'data/output/{args.task}/{args.method_name}/{args.data_split}/')
        
        if args.subset is not None:
            qa_pred_path = output_dir / args.subset / args.model_name / args.template_name / f'data_{args.question_num}.jsonl'
        else:
            qa_pred_path = output_dir / args.model_name / args.template_name / f'data_{args.question_num}.jsonl'
        # breakpoint()

        if args.template_name.startswith('answer_generation') or args.template_name.startswith('qa_pair_generation'):
            if args.task == 'radqa':
                data_generated_json = process_generated_qa_radqa(qa_pred_path, args.qa_select_num)
            elif args.task == 'mimicqa':
                data_generated_json = process_generated_qa_mimicqa(qa_pred_path, args.qa_select_num)
            else:
                raise ValueError(f"Task {args.task} not recognized")
                
        elif args.template_name.startswith('question_paraphrase'):
            data_generated_json = process_paraphrased_questions(qa_pred_path)
        else:
            raise ValueError(f"Template name {args.template_name} not recognized")
        
        output_path_json = qa_pred_path.parent.parent / f'data_generate_{args.question_num}_select_{args.qa_select_num}.json'
        print(f"Saving json format to {output_path_json}")
        
        with open(output_path_json, 'w') as f:
            json.dump(data_generated_json, f)
    else:
        output_path_json = Path(args.output_path_json)
        print(f"Reading from {output_path_json}")
        with open(output_path_json) as f:
            data_generated_json = json.load(f)
    
    data_generated_csv = convert_squad_to_csv_fmt(data_generated_json)
    
    if args.output_path_json is None:
        output_path_csv = qa_pred_path.parent.parent / f'data_generate_{args.question_num}_select_{args.qa_select_num}.csv'
    else:
        output_path_csv = output_path_json.parent / f'{output_path_json.stem}.csv'
    print(f"Saving csv format to {output_path_csv}")

    csv = pd.DataFrame(data_generated_csv)    
    csv.to_csv(output_path_csv, index=False)
    
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='radqa', required=True)
    parser.add_argument('--data_split', type=str, default='train')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--method_name', type=str)
    parser.add_argument('--template_name', type=str)
    parser.add_argument('--subset', type=str)
    parser.add_argument('--question_num', type=int)
    parser.add_argument('--qa_select_num', type=int, required=True)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--output_path_json', type=str)
    
    args = parser.parse_args()
    main(args)