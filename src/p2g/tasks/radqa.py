import re
import os
import json
import sympy
import ast
import time
import random

import pandas as pd
import numpy as np

from collections import Counter, defaultdict

from p2g.tasks.base import Task, DATA_PATH
from p2g.methods.prompt import *
from p2g.templates.radqa import * 

# from openai.embeddings_utils import cosine_similarity

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class RadQAItem:
    def __init__(self, para_item: dict):
        self.context = para_item['context']
        self.qas = para_item['qas']
        self.document_id = para_item['document_id']
        self.prompt = para_item.get('prompt', None)
        self.messages = para_item.get('messages', None)
        self.summary = para_item.get('summary', None)
        self.questions_generated = para_item.get('questions_generated', None)
        self.qas_generated = para_item.get('qas_generated', None)
        self.output = para_item.get('output', None)
        
    def __repr__(self):
        return self.document_id
    
    def prompt_wrap(self, template_name: str, template: str, question_num: int = 1, paraphrase_num: int = 1):
        
        if template_name.startswith('question_generation'):
            if template_name.startswith('question_generation_plain'):
                assert "{{input_context}}" in template
                prompt = template.replace("{{input_context}}", self.context)
            elif template_name.startswith('question_generation_summary'):
                assert "{{input_summary}}" in template
                prompt = template.replace("{{input_summary}}", self.summary)
                # breakpoint()
                
            assert "{{question_num}}" in template
            prompt = prompt.replace("{{question_num}}", str(question_num))
            
        elif template_name.startswith('answer_generation'):
            assert "{{input_context}}" in template
            assert "{{input_questions}}" in template
            assert self.questions_generated is not None
            prompt = template.replace("{{input_context}}", self.context)
            prompt = prompt.replace("{{input_questions}}", self.questions_generated)
            # breakpoint()
        elif template_name.startswith('summary_generation'):
            assert "{{input_context}}" in template
            prompt = template.replace("{{input_context}}", self.context)
        else:
            raise ValueError(f"Unrecognized template '{template_name}' for RadQA")
        self.prompt = prompt
        self.messages = [
            {"role": "user", "content": prompt}
        ]
    
    def prompt_wrap_icl(self, template_name: str, template: str, icl_examples_str: str):
        assert "{{icl_examples}}" in template
        prompt = template.replace("{{icl_examples}}", icl_examples_str)
        
        assert "{{input_context}}" in prompt
        prompt = prompt.replace("{{input_context}}", self.context)
        self.prompt = prompt
        self.messages = [
            {"role": "user", "content": prompt}
        ]

    def update_output(self, template_name: str, response: str):
        
        if template_name.startswith('question_generation'):
            if "Empty input" in response:
                self.questions_generated = None
            else:
                self.questions_generated = self.extract_questions(response)
        elif template_name.startswith('answer_generation'):
            if response is not None:
                qa_list = [item for item in response.strip().split('\n\n') if item.startswith('Q:')]
            else:
                qa_list = [re.sub(r'^\d+[\.:]\s*', 'Q: ', q)+'\n'+'A: Unanswerable' for q in self.questions_generated.split('\n')]
            self.qas_generated = '\n\n'.join(qa_list)
        elif template_name.startswith('summary_generation'):
            self.summary = self.extract_json(response)
            # breakpoint()
        else:
            raise ValueError(f"Unrecognized template '{template_name}' for RadQA")
        
    def process_qa_generated(self):
        
        qas_generated = self.qas_generated.split('\n\n')
        qas_generated_processed = []
        
        for qa_idx, qa_item in enumerate(qas_generated):
            
            if len(qa_item.split('\n')) != 2:
                continue
                
            q_str, a_str = qa_item.split('\n')
            if not q_str.startswith('Q:'):
                continue
            
            if not a_str.startswith('A:'):
                continue
            
            q_str = q_str[2:].strip()
            a_str = a_str[2:].strip().strip('"').strip()
            
            q_str = self.remove_prefix(q_str)
            if not q_str or not a_str:
                continue
            
            q_id = f"{self.document_id}_Q{qa_idx}"
            a_id = f"{self.document_id}_Q{qa_idx}_A{qa_idx}"
            
            if a_str.startswith("Unanswerable"):
                a_str = 'Unanswerable'
            else:
                if a_str in self.context:
                    a_str = self.process_answers(a_str)
                    
                    if not a_str:
                        continue
                    
                    assert a_str in self.context
                else:
                    continue
                
            qas_generated_processed.append(
                {'q_id': q_id, 'question': q_str, 'a_id': a_id, 'answer': a_str}
            )
            
        return qas_generated_processed
                    
    @staticmethod
    def process_answers(answer_span):
        answer_span = answer_span.strip()
        answer_span = answer_span.strip('.').strip()
        
        if 'there' in answer_span.lower():
            answer_span = answer_span[answer_span.lower().index('there'):].strip()
        
        if answer_span.lower().startswith('there'):
            
            # Replace 'There is' on the left with an empty string
            for starter_phrase in ['There is', 'There are', 'There was', 'There were', 'there is', 'there are', 'there was', 'there were']:
                if answer_span.startswith(starter_phrase):
                    answer_span = answer_span[len(starter_phrase):].strip()
                    break
        
        return answer_span
            
    @staticmethod
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
        
    @staticmethod
    def extract_questions(response):
        
        # Preprocess the question index, from "**1.**" to "1."
        response = re.sub(r'\*\*(\d+)\.\*\*', r'\1.', response)
    
        # Regular expression to find a JSON-like object
        pattern = r'^\d+\.\s.*\?'
        
        # Find all matches with multiline flag enabled
        questions = re.findall(pattern, response, re.M)
        if len(questions) < 1:
            print(f"Expected at least 1 match, got {len(questions)} from response below:\n{response}")
        
        return '\n'.join(questions)
    
    @staticmethod
    def extract_json(response):
    
        # Regular expression to find a JSON-like object
        pattern = r'\{(?:[^{}]*|\{[^{}]*\})*\}'
        
        # Find all matches
        matches = re.findall(pattern, response)
        assert len(matches) >= 1
        
        if len(matches) > 1:
            print(f"Expected 1 match, got {len(matches)} from {response}")
        
        # return json.loads(matches[0])
        return matches[0]
            
                
class RadQAPairItem:
    def __init__(self, id:str, question: str, answers: list, context: str):
        self.id = id
        self.question = question
        self.answer = answers
        self.context = context
        self.prompt = None
        self.messages = None
        self.output = None
        
    def __repr__(self):
        return self.question
    
    def prompt_wrap(self, template_name: str, template: str, question_num: int = 1, paraphrase_num: int = 1):
        if template_name.startswith('run_qa_'):
            assert "{{input_context}}" in template
            prompt = template.replace("{{input_context}}", self.context)
            
            assert "{{input_question}}" in prompt
            prompt = prompt.replace("{{input_question}}", self.question)
        elif template_name.startswith('generate_question_'):
            assert "{{input_context}}" in template
            prompt = template.replace("{{input_context}}", self.context)
            
            assert "{{input_answer}}" in prompt
            prompt = prompt.replace("{{input_answer}}", self.answer[0]['text'])
        elif template_name.startswith('qa_zs'):
            assert "{{input_context}}" in template
            assert "{{input_question}}" in template
            assert self.question is not None
            prompt = template.replace("{{input_context}}", self.context)
            prompt = prompt.replace("{{input_question}}", self.question)
        elif template_name.startswith('question_paraphrase'):
            assert "{{input_context}}" in template
            assert "{{input_question}}" in template
            assert "{{input_answer}}" in template
            assert "{{paraphrase_num}}" in template
            prompt = template.replace("{{input_context}}", self.context)
            prompt = prompt.replace("{{input_question}}", self.question)
            prompt = prompt.replace("{{input_answer}}", self.answer[0]['text'])
            prompt = prompt.replace("{{paraphrase_num}}", str(paraphrase_num))
        else:
            raise ValueError(f"Unrecognized template '{template_name}' for RadQA")
        self.prompt = prompt
        self.messages = [
            {"role": "user", "content": prompt}
        ]
        
    def update_output(self, template_name: str, response: str):
        self.output = response
        
        
class RadQADocItem:    
    def __init__(self, doc_item: list):
        self.id = doc_item['title'] if 'title' in doc_item else doc_item['id']
        
        if 'paragraphs' in doc_item:
            paragraphs = doc_item['paragraphs']
            assert len(paragraphs) == 2, f"Expected 2 paragraphs, got {len(paragraphs)}"
            assert paragraphs[0]['document_id'] == f'{self.id}_I'
            assert paragraphs[1]['document_id'] == f'{self.id}_O'
            self.context_impression = paragraphs[0]['context']
            self.context_findings = paragraphs[1]['context']
            self.context = f'{self.context_findings}\n\n{self.context_impression}'
        else:
            self.context_impression = doc_item['context_impression']
            self.context_findings = doc_item['context_findings']
            self.context = doc_item['context']
        
        self.prompt = doc_item.get('prompt', None)
        self.response = None
        self.messages = doc_item.get('messages', None)
        self.schema_generated = doc_item.get('schema_generated', None)
        self.summary = doc_item.get('summary', None)
        self.summary_know = doc_item.get('summary_know', None)
        self.questions_generated = doc_item.get('questions_generated', None)
        self.qas_generated = doc_item.get('qas_generated', None)
        self.output = doc_item.get('output', None)
        # breakpoint()
        
    def prompt_wrap(self, template_name: str, template: str, question_num: int = 1, paraphrase_num: int = 1):
    
        if template_name.startswith('question_generation'):
            if template_name in ['question_generation_plain', 'question_generation_explicit', 'question_generation_nonoverlap']:
                assert "{{input_context}}" in template
                prompt = template.replace("{{input_context}}", self.context)
            elif template_name in ['question_generation_summary', 'question_generation_summary_nonoverlap']:
                assert "{{input_summary}}" in template
                
                if self.summary == "No output due to Azure OpenAI's content management.":
                    prompt = template.replace("{{input_summary}}", "Chest pain")
                else:
                    prompt = template.replace("{{input_summary}}", self.summary)
            elif template_name in ['question_generation_summary_know']:
                assert "{{input_summary}}" in template
                assert "{{external_knowledge}}" in template
                
                if self.summary is None or self.summary == "No output due to Azure OpenAI's content management.":
                    prompt = template.replace("{{input_summary}}", "Chest pain")
                else:
                    prompt = template.replace("{{input_summary}}", self.summary)
                    
                if self.summary_know is None or self.summary_know == "No output due to Azure OpenAI's content management.":
                    prompt = prompt.replace("{{external_knowledge}}", "Chest pain")
                else:
                    prompt = prompt.replace("{{external_knowledge}}", self.summary_know)
            elif template_name in ['question_generation_summary_know_only']:
                assert "{{external_knowledge}}" in template
                
                if self.summary_know is None or self.summary_know == "No output due to Azure OpenAI's content management.":
                    prompt = template.replace("{{external_knowledge}}", "Chest pain")
                else:
                    prompt = template.replace("{{external_knowledge}}", self.summary_know)
            else:
                raise ValueError(f"Unrecognized template '{template_name}' for RadQA")
            
            assert "{{question_num}}" in template
            prompt = prompt.replace("{{question_num}}", str(question_num))
            
        elif template_name.startswith('summary_generation'):
            assert "{{input_context}}" in template
            prompt = template.replace("{{input_context}}", self.context)
        elif template_name.startswith('schema_generation'):
            prompt = template
        elif template_name.startswith('knowledge_elicitation'):
            assert "{{input_summary}}" in template
            
            if self.summary == "No output due to Azure OpenAI's content management.":
                prompt = template.replace("{{input_summary}}", "Chest pain")
            else:
                prompt = template.replace("{{input_summary}}", self.summary)
            
        else:
            raise ValueError(f"Unrecognized template '{template_name}' for RadQA")
        self.prompt = prompt
        self.messages = [
            {"role": "user", "content": prompt}
        ]
        # breakpoint()
        
    def update_output(self, template_name: str, response: str):
        self.response = response
        
        if template_name.startswith('question_generation'):
            if response is None or "Empty input" in response:
                self.questions_generated = None
            else:
                self.questions_generated = self.extract_questions(response)
            # breakpoint()
        elif template_name.startswith('summary_generation'):
            if template_name.endswith('_json'):
                self.summary = self.extract_json(response)
            elif template_name.endswith('_md'):
                response_processed = [item for item in response.split('\n\n') if not item.startswith('Note: ')]
                self.summary = '\n'.join(response_processed)
            else:
                self.summary = response
            # breakpoint()
        elif template_name.startswith('knowledge_elicitation'):
            self.summary_know = response
        elif template_name.startswith('schema_generation'):
            self.schema_generated = self.extract_json(response)    
        else:
            raise ValueError(f"Unrecognized template '{template_name}' for RadQA")
        
    @staticmethod
    def extract_questions(response):
    
        # Preprocess the question index, from "**1.**" to "1."
        response = re.sub(r'\*\*(\d+)\.\*\*', r'\1.', response)
    
        # Regular expression to find a JSON-like object
        pattern = r'^\d+\.\s.*\?'
        
        # Find all matches with multiline flag enabled
        questions = re.findall(pattern, response, re.M)
        if len(questions) < 1:
            print(f"Expected at least 1 match, got {len(questions)} from response below:\n{response}")
        
        return '\n'.join(questions)
        
    @staticmethod
    def extract_json(response):
    
        # Regular expression to find a JSON-like object
        pattern = r'\{(?:[^{}]*|\{[^{}]*\})*\}'
        
        # Find all matches
        matches = re.findall(pattern, response)
        assert len(matches) >= 1
        
        if len(matches) > 1:
            print(f"Expected 1 match, got {len(matches)} from {response}")
        
        # return json.loads(matches[0])
        return matches[0]
   
class RadQA(Task):
    def __init__(self, args) -> None:

        self.args = args
        self.template = globals()[args.template]
        
        if args.prior_template is None:
            self.data_path = self.get_data_path(args, args.data_split)
        else:
            self.data_path = self.get_output_data_path(args.prior_template)
        # breakpoint()
            
        self.data = self.load_data(self.data_path, args.task_start_index, args.task_end_index)
        print(f"Load {len(self.data)} {self.args.process_unit} from {self.data_path}")
        # breakpoint()
        
        if args.prompt_setting == 'icl':
            self.icl_data_path = self.get_data_path(args, 'train')
            self.icl_data = self.load_data(self.icl_data_path, args.icl_start_index, args.icl_end_index)
        
        self.data_wrap()
        
    def get_data_path(self, args, data_split: str) -> str:
        if not args.answerable_only and not args.unanswerable_only and not args.data_subset:
            # data_path = os.path.join(DATA_PATH, 'modified', self.args.task, f'{data_split}.json')
            data_path = os.path.join(DATA_PATH, 'modified', self.args.task, f'{data_split}_processed.json')
        else:
            if args.answerable_only:
                data_path = os.path.join(DATA_PATH, 'modified', args.task, 'answerable', f'{data_split}.json' if not args.data_subset else f'{data_split}_{args.data_subset}.json')
            elif args.unanswerable_only:
                data_path = os.path.join(DATA_PATH, 'modified', args.task, 'unanswerable', f'{data_split}.json' if not args.data_subset else f'{data_split}_{args.data_subset}.json')
            else:
                data_path = os.path.join(DATA_PATH, 'modified', args.task, f'{data_split}.json' if not args.data_subset else f'{data_split}_{args.data_subset}.json')
        return data_path
        
    def load_data(self, data_path:str, start_index: int, end_index: int) -> list:
        
        if self.args.prior_template is None:
            with open(data_path) as f:
                data = json.load(f)
                
            if self.args.process_unit == 'paragraph':
                data = [RadQAItem(paragraph) for item in data['data'] for paragraph in item['paragraphs']]
            elif self.args.process_unit == 'qa_pair':
                data = [RadQAPairItem(qa['id'], qa['question'], qa['answers'], paragraph['context']) for item in data['data'] for paragraph in item['paragraphs'] for qa in paragraph['qas']]
            elif self.args.process_unit == 'document':
                data = [RadQADocItem(item) for item in data['data']]
            else:
                raise ValueError(f"Unrecognized process unit '{self.args.process_unit}' for RadQA")
            
        else:
            with open(data_path) as f:
                data = [json.loads(line) for line in f]
                
            if self.args.prior_process_unit is None or self.args.prior_process_unit == self.args.process_unit:
                if self.args.process_unit == 'paragraph':
                    data = [RadQAItem(para_item) for para_item in data]
                elif self.args.process_unit == 'document':
                    data = [RadQADocItem(doc_item) for doc_item in data]
                else:
                    raise ValueError(f"Unrecognized process unit '{self.args.process_unit}' for RadQA")
            else:
                if self.args.prior_process_unit == 'paragraph':
                    data_prior = [RadQAItem(para_item) for para_item in data]
                elif self.args.prior_process_unit == 'document':
                    data_prior = [RadQADocItem(doc_item) for doc_item in data]
                else:
                    raise ValueError(f"Unrecognized prior process unit '{self.args.prior_process_unit}' for RadQA")
                
                if self.args.prior_process_unit == 'document' and self.args.process_unit == 'paragraph':
                    data = []
                    for idx in range(0, len(data_prior)):
                        data.append({'document_id': f'{data_prior[idx].id}_I', 'context': data_prior[idx].context_impression, 'qas': None, 'questions_generated': data_prior[idx].questions_generated})
                        data.append({'document_id': f'{data_prior[idx].id}_O', 'context': data_prior[idx].context_findings, 'qas': None, 'questions_generated': data_prior[idx].questions_generated})
                    assert len(data) == 2*len(data_prior)
                    data = [RadQAItem(para_item) for para_item in data]
                elif self.args.prior_process_unit == 'paragraph' and self.args.process_unit == 'qa_pair':
                    data = []
                    for idx in range(0, len(data_prior)):
                        qas_generated_processed = data_prior[idx].process_qa_generated()
                        for qa in qas_generated_processed:
                            data.append({'id': qa['q_id'], 'question': qa['question'], 'answers': [{'text': qa['answer']}], 'context': data_prior[idx].context})
                    data = [RadQAPairItem(qa['id'], qa['question'], qa['answers'], qa['context']) for qa in data]
                else:
                    raise ValueError(f"Unrecognized prior process unit '{self.args.prior_process_unit}' and process unit '{self.args.process_unit}' for RadQA")
                            
        if end_index > 0:
            if self.args.prior_process_unit == 'document' and self.args.process_unit == 'paragraph':
                return data[start_index*2:end_index*2]
            if self.args.prior_process_unit == 'paragraph' and self.args.process_unit == 'qa_pair':
                return data
            else:
                return data[start_index:end_index]
        else:
            return data
        
    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        # return self.data[idx].prompt
        return self.data[idx].messages
    
    def get_input_list(self) -> list:
        return [self.get_input(idx) for idx in range(len(self.data))]
    
    def get_data(self) -> list:
        return self.data
    
    def get_icl_examples(self, item_idx, item) -> list:
        if self.args.icl_style == 'random':
            random.seed(self.args.icl_random_seed)
            shuffle_idx = list(range(len(self.icl_data)))
            random.shuffle(shuffle_idx)
            shuffle_idx = shuffle_idx[:self.args.icl_num]
            return [self.icl_data[idx] for idx in shuffle_idx]
        elif self.args.icl_style == 'retrieval':
            pass
        else:
            raise ValueError(f"Unrecognized icl style '{self.args.icl_style}' for RadQA")
        
    def build_icl_examples_str(self, icl_examples) -> str:
        icl_examples_processed = []
        for item in icl_examples:
            context_str = item.context
            qas = item.qas
            qa_pairs_str = "\n\n".join([f"Question: {qa['question']}\nAnswer: {qa['answers'][0]['text']}" for qa in qas])
            item_str = f"Context: {context_str}\n\n{qa_pairs_str}"
            # print(item_str)

            icl_examples_processed.append(item_str)
        icl_examples_str = '\n\n'.join(icl_examples_processed)
        # print(icl_examples_str)

        return icl_examples_str
    
    def data_wrap(self) -> None:
        if self.args.prompt_setting == 'naive':
            for item in self.data:
                item.prompt_wrap(self.args.template, self.template, self.args.question_num, self.args.paraphrase_num)
        elif self.args.prompt_setting == 'icl':
            for item_idx, item in enumerate(self.data):
                icl_examples = self.get_icl_examples(item_idx, item)
                icl_examples_str = self.build_icl_examples_str(icl_examples)
                item.prompt_wrap_icl(self.args.template, self.template, icl_examples_str)
        else:
            raise ValueError(f"Prompt setting {self.args.prompt_setting} not supported")
        
    def update_output(self, responses: list) -> list:
        assert len(self.data) == len(responses)
        for item, response in zip(self.data, responses):
            item.update_output(self.args.template, response)
        
    def update_output_icl(self, responses: list) -> list:
        # output_prefix = self.template.split("\n")[-1]
        
        for item, response in zip(self.data, responses):
            # try: 
            #     if not response.startswith(output_prefix):
            #         response = output_prefix+response
            #     else:
            #         response = response
            #     response = response.strip()
                
            # except Exception as e:
            #     print(f"Error {e} when processing {output_prefix+response}")
            #     response = ""
            
            item.update_output(self.args.template, response)
            
    def get_output_data_path(self, template: str) -> str:
        
        if self.args.api_source == 'openai':
            model_name = self.args.backend
        elif self.args.api_source == 'azure':
            model_name = AZURE_MODELS[self.args.backend]
        elif self.args.api_source == 'open_source':
            model_name = self.args.backend
        else:
            raise ValueError(f'Unknown api source: {self.args.api_source}')
        
        output_dir = os.path.join(DATA_PATH, 'output', self.args.task, self.args.method_name, self.args.data_split)
            
        if self.args.task_end_index > 0:
            output_dir = os.path.join(output_dir, f'{self.args.task_start_index}_{self.args.task_end_index}', model_name, template)
        else:
            output_dir = os.path.join(output_dir, model_name, template)
        # breakpoint()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        if template.startswith('question_generation') or template.startswith('answer_generation'):
            file_name = f'data_{self.args.question_num}.jsonl'
        else:
            file_name = 'data.jsonl'
        output_path = os.path.join(output_dir, file_name)
        return output_path
            
    def save_output(self) -> None:

        output_data_path = self.get_output_data_path(self.args.template)
        # breakpoint()
        
        output_args_path = output_data_path.replace('data.jsonl', 'args.json')
        
        with open(output_args_path, 'w') as f:
            json.dump(self.args.__dict__, f)
        
        with open(output_data_path, 'w') as f:
            for item in self.data:
                # Dump the RadQAItem object as a json string
                if self.args.template.startswith('question_generation'):
                    if item.questions_generated is None:
                        continue
                    
                json.dump(item.__dict__, f)
                f.write('\n')
            
    def save_output_icl(self) -> None:
        
        if self.args.api_source == 'openai':
            model_name = self.args.backend
        elif self.args.api_source == 'azure':
            model_name = AZURE_MODELS[self.args.backend]
        elif self.args.api_source == 'open_source':
            model_name = self.args.backend
        else:
            raise ValueError(f'Unknown api source: {self.args.api_source}')
        
        if self.args.answerable_only:
            output_dir = os.path.join(DATA_PATH, 'output', self.args.task, 'answerable', self.args.data_split, model_name, self.args.template)
        elif self.args.unanswerable_only:
            output_dir = os.path.join(DATA_PATH, 'output', self.args.task, 'unanswerable', self.args.data_split, model_name, self.args.template) 
        else:
            output_dir = os.path.join(DATA_PATH, 'output', self.args.task, self.args.data_split, model_name, self.args.template) 
            
        if self.args.data_subset:
            output_dir = os.path.join(output_dir, self.args.data_subset)
            
        if self.args.task_start_index != -1:
            output_dir = os.path.join(output_dir, f'{self.args.task_start_index}_{self.args.task_end_index}')
        # breakpoint()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file_name = f'{self.args.data_split}_context_{self.args.icl_num}' if self.args.icl_start_index == -1 else f'train_{self.args.icl_start_index}_{self.args.icl_end_index}_context_{self.args.icl_num}'
        file_name = f'{file_name}_random_{self.args.icl_random_seed}' if self.args.icl_style == 'random' else f'{file_name}_retrieval_{self.args.retrieval_model}'
        file_name = f'{file_name}.jsonl'

        output_path = os.path.join(output_dir, file_name)
        # breakpoint()

        with open(output_path, 'w') as f:
            for item in self.data:
                # Dump the RadQAItem object as a json string
                json.dump(item.__dict__, f)
                f.write('\n')
                