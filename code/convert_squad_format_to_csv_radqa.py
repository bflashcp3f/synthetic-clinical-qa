import json
import pandas as pd

def convert_squad_to_csv_fmt(jf: dict, out_f: str):
    """Convert file to csv format. """
    new_data = {'id': [], 'title': [], 'context': [], 'question': [], 'answers': []}    

    data = jf['data']
    for note in data:
        title = note['title']
        for para in note['paragraphs']:
            context = para['context']

            for qas in para['qas']:
                new_data['id'].append(qas['id'])
                new_data['title'].append(title)
                new_data['context'].append(context)
                new_data['question'].append(qas['question'])
                
                if qas['is_impossible']:
                    new_data['answers'].append({'text': [], 'answer_start': []})
                    continue 
                
    
                to_add = {'text': [], 'answer_start': []}
                for a in qas['answers']:
                    to_add['text'].append(a['text'])
                    to_add['answer_start'].append(a['answer_start'])
    
                new_data['answers'].append(to_add)
    
    csv = pd.DataFrame(new_data)
    csv.to_csv(out_f, index=False)

if __name__ == '__main__':
    # pairs = [
    #     ('data/raw/radqa/train.json', 'data/modified/radqa/train.csv'),
    #     ('data/raw/radqa/dev.json', 'data/modified/radqa/dev.csv'),
    #     ('data/raw/radqa/test.json', 'data/modified/radqa/test.csv')
    # ]
    
    pairs = [
        # ('data/modified/radqa/answerable/train.json', 'data/modified/radqa/answerable/train.csv'),
        # ('data/modified/radqa/answerable/dev.json', 'data/modified/radqa/answerable/dev.csv'),
        # ('data/modified/radqa/answerable/test.json', 'data/modified/radqa/answerable/test.csv'),
        
        # ('data/modified/radqa/answerable/train_findings.json', 'data/modified/radqa/answerable/train_findings.csv'),
        # ('data/modified/radqa/answerable/dev_findings.json', 'data/modified/radqa/answerable/dev_findings.csv'),
        # ('data/modified/radqa/answerable/test_findings.json', 'data/modified/radqa/answerable/test_findings.csv'),
        
        # ('data/modified/radqa/answerable/train_impressions.json', 'data/modified/radqa/answerable/train_impressions.csv'),
        # ('data/modified/radqa/answerable/dev_impressions.json', 'data/modified/radqa/answerable/dev_impressions.csv'),
        # ('data/modified/radqa/answerable/test_impressions.json', 'data/modified/radqa/answerable/test_impressions.csv'),
        
        # ('data/modified/radqa/answerable/train_findings_llama2chat_random_42.json', 'data/modified/radqa/answerable/train_findings_llama2chat_random_42.csv'),
        # ('data/modified/radqa/answerable/dev_findings_llama2chat_random_42.json', 'data/modified/radqa/answerable/dev_findings_llama2chat_random_42.csv'),
        # ('data/modified/radqa/answerable/test_findings_llama2chat_random_42.json', 'data/modified/radqa/answerable/test_findings_llama2chat_random_42.csv'),
        
        # ('data/modified/radqa/answerable/train_findings_llama2chat_random_1234.json', 'data/modified/radqa/answerable/train_findings_llama2chat_random_1234.csv'),
        # ('data/modified/radqa/answerable/train_findings_llama2chat_random_2333.json', 'data/modified/radqa/answerable/train_findings_llama2chat_random_2333.csv'),
        # ('data/modified/radqa/answerable/train_findings_llama2chat_random_all.json', 'data/modified/radqa/answerable/train_findings_llama2chat_random_all.csv'),
        
        # ('data/modified/radqa/answerable/train_findings_llama2chat_complex_random_42.json', 'data/modified/radqa/answerable/train_findings_llama2chat_complex_random_42.csv'),
        # ('data/modified/radqa/answerable/dev_findings_llama2chat_complex_random_42.json', 'data/modified/radqa/answerable/dev_findings_llama2chat_complex_random_42.csv'),
        # ('data/modified/radqa/answerable/test_findings_llama2chat_complex_random_42.json', 'data/modified/radqa/answerable/test_findings_llama2chat_complex_random_42.csv'),
        
        # ('data/modified/radqa/answerable/train_findings_generate_question_top_1_llama2chat.json', 'data/modified/radqa/answerable/train_findings_generate_question_top_1_llama2chat.csv'),
        # ('data/modified/radqa/answerable/train_findings_generate_question_top_2_llama2chat.json', 'data/modified/radqa/answerable/train_findings_generate_question_top_2_llama2chat.csv'),
        
        # ('data/modified/radqa/answerable/train_findings_sample_10_random_42.json', 'data/modified/radqa/answerable/train_findings_sample_10_random_42.csv'),
        # ('data/modified/radqa/answerable/train_findings_sample_20_random_42.json', 'data/modified/radqa/answerable/train_findings_sample_20_random_42.csv'),
        # ('data/modified/radqa/answerable/train_findings_sample_30_random_42.json', 'data/modified/radqa/answerable/train_findings_sample_30_random_42.csv'),
        
        # ('data/modified/radqa/answerable/train_findings_generate_context_qa_chatgpt_10_random_42.json', 'data/modified/radqa/answerable/train_findings_generate_context_qa_chatgpt_10_random_42.csv'),
        # ('data/modified/radqa/answerable/train_findings_generate_context_qa_gold_gpt4_sample_10_random_42.json', 'data/modified/radqa/answerable/train_findings_generate_context_qa_gold_gpt4_sample_10_random_42.csv'),
        # ('data/modified/radqa/answerable/train_findings_generate_context_qa_pred_gpt4_sample_10_random_42.json', 'data/modified/radqa/answerable/train_findings_generate_context_qa_pred_gpt4_sample_10_random_42.csv'),
        # ('data/modified/radqa/answerable/train_findings_generate_context_qa_pred_para_gpt4_sample_10_random_42.json', 'data/modified/radqa/answerable/train_findings_generate_context_qa_pred_para_gpt4_sample_10_random_42.csv'),
        # ('data/modified/radqa/answerable/train_findings_generate_context_qa_pred_para_combined_gpt4_sample_10_random_42.json', 'data/modified/radqa/answerable/train_findings_generate_context_qa_pred_para_combined_gpt4_sample_10_random_42.csv'),
        
        # ('data/modified/radqa/answerable/train_findings_generate_context_qa_pred_gpt4_sample_10_random_42_sample_qa_random_0.json', 'data/modified/radqa/answerable/train_findings_generate_context_qa_pred_gpt4_sample_10_random_42_sample_qa_random_0.csv'),
        # ('data/modified/radqa/answerable/train_findings_generate_context_qa_pred_gpt4_sample_10_random_42_sample_qa_random_1.json', 'data/modified/radqa/answerable/train_findings_generate_context_qa_pred_gpt4_sample_10_random_42_sample_qa_random_1.csv'),
        # ('data/modified/radqa/answerable/train_findings_generate_context_qa_pred_gpt4_sample_10_random_42_sample_qa_random_2.json', 'data/modified/radqa/answerable/train_findings_generate_context_qa_pred_gpt4_sample_10_random_42_sample_qa_random_2.csv'),
        
        # ('data/modified/radqa/answerable/train_findings_generate_context_qa_gold_augment_gpt4_sample_10_random_42.json', 'data/modified/radqa/answerable/train_findings_generate_context_qa_gold_augment_gpt4_sample_10_random_42.csv'),
        
        # ('data/modified/radqa/answerable_syn_context/Llama-2-7b-chat-hf/formatted/train_findings_generate_context_qa_pred_sample_10_random_42.json', 'data/modified/radqa/answerable_syn_context/Llama-2-7b-chat-hf/formatted/train_findings_generate_context_qa_pred_sample_10_random_42.csv'),
        # ('data/modified/radqa/answerable_syn_context/Llama-2-70b-chat-hf/formatted/train_findings_generate_context_qa_pred_sample_10_random_42.json', 'data/modified/radqa/answerable_syn_context/Llama-2-70b-chat-hf/formatted/train_findings_generate_context_qa_pred_sample_10_random_42.csv'),
        # ('data/modified/radqa/answerable_syn_context/Mixtral-8x7B-Instruct-v0.1/formatted/train_findings_generate_context_qa_pred_sample_10_random_42.json', 'data/modified/radqa/answerable_syn_context/Mixtral-8x7B-Instruct-v0.1/formatted/train_findings_generate_context_qa_pred_sample_10_random_42.csv'),
        
        # ('data/modified/radqa/answerable_syn_context/Llama-2-7b-chat-hf/formatted/train_findings_generate_context_qa_gold_augment_sample_10_random_42.json', 'data/modified/radqa/answerable_syn_context/Llama-2-7b-chat-hf/formatted/train_findings_generate_context_qa_gold_augment_sample_10_random_42.csv'),
        # ('data/modified/radqa/answerable_syn_context/Llama-2-70b-chat-hf/formatted/train_findings_generate_context_qa_gold_augment_sample_10_random_42.json', 'data/modified/radqa/answerable_syn_context/Llama-2-70b-chat-hf/formatted/train_findings_generate_context_qa_gold_augment_sample_10_random_42.csv'),
        # ('data/modified/radqa/answerable_syn_context/Mixtral-8x7B-Instruct-v0.1/formatted/train_findings_generate_context_qa_gold_augment_sample_10_random_42.json', 'data/modified/radqa/answerable_syn_context/Mixtral-8x7B-Instruct-v0.1/formatted/train_findings_generate_context_qa_gold_augment_sample_10_random_42.csv'),
        
        ('data/modified/radqa/answerable_syn_context/generate-direct/gpt-4-turbo-2024-04-09/formatted/train_findings_generate_context_qa_pred_sample_10_random_42.json', 'data/modified/radqa/answerable_syn_context/generate-direct/gpt-4-turbo-2024-04-09/formatted/train_findings_generate_context_qa_pred_sample_10_random_42.csv'),
        
        ('data/modified/radqa/answerable_syn_context/generate-direct/gpt-4-turbo-2024-04-09/formatted/train_findings_generate_context_qa_gold_augment_sample_10_random_42.json', 'data/modified/radqa/answerable_syn_context/generate-direct/gpt-4-turbo-2024-04-09/formatted/train_findings_generate_context_qa_gold_augment_sample_10_random_42.csv'),
        
        ('data/modified/radqa/answerable_syn_context/generate-direct_simple/gpt-4-turbo-2024-04-09/formatted/train_findings_generate_context_qa_pred_sample_10_random_42.json', 'data/modified/radqa/answerable_syn_context/generate-direct_simple/gpt-4-turbo-2024-04-09/formatted/train_findings_generate_context_qa_pred_sample_10_random_42.csv'),
        
        # ('data/modified/radqa/answerable_syn_context/generate-direct_simple/gpt-4-turbo-2024-04-09/formatted/train_findings_generate_context_qa_gold_augment_sample_10_random_42.json', 'data/modified/radqa/answerable_syn_context/generate-direct_simple/gpt-4-turbo-2024-04-09/formatted/train_findings_generate_context_qa_gold_augment_sample_10_random_42.csv'),
        
        # ('data/modified/radqa/train_processed.json', 'data/modified/radqa/train_processed.csv'),
        # ('data/modified/radqa/dev_processed.json', 'data/modified/radqa/dev_processed.csv'),
        # ('data/modified/radqa/test_processed.json', 'data/modified/radqa/test_processed.csv'),
        
        # ('data/modified/radqa/train_100_processed.json', 'data/modified/radqa/train_100_processed.csv'),
        
        ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_report_simple/0_100/train_processed.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_report_simple/0_100/train_processed.csv'),
        ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_report_complex/0_100/train_processed.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_report_complex/0_100/train_processed.csv'),
        ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_simple_summary_generation/0_100/train_processed.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_simple_summary_generation/0_100/train_processed.csv'),
        ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_complex_summary_generation/0_100/train_processed.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_complex_summary_generation/0_100/train_processed.csv'),
        
        # ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_report_simple/train_processed.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_report_simple/train_processed.csv'),
        # ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_report_complex/train_processed.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_report_complex/train_processed.csv'),
        # ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_simple_summary_generation/train_processed.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_simple_summary_generation/train_processed.csv'),
        # ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_complex_summary_generation/train_processed.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_complex_summary_generation/train_processed.csv'),
        
        # ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_report_simple/train_processed_sampled.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_report_simple/train_processed_sampled.csv'),
        # ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_simple_summary_generation/train_processed_sampled.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_simple_summary_generation/train_processed_sampled.csv'),
        
        # ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_report_simple/train_processed_answerable.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_report_simple/train_processed_answerable.csv'),
        # ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_simple_summary_generation/train_processed_answerable.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_simple_summary_generation/train_processed_answerable.csv'),
        # ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_report_simple/train_processed_sampled_answerable.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_report_simple/train_processed_sampled_answerable.csv'),
        # ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_simple_summary_generation/train_processed_sampled_answerable.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_simple_summary_generation/train_processed_sampled_answerable.csv'),
        
        # ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_chatgpt_summary_generation_cxt_both/train_processed.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_chatgpt_summary_generation_cxt_both/train_processed.csv'),
        # ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_chatgpt_summary_generation_cxt_both/train_processed_answerable.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_chatgpt_summary_generation_cxt_both/train_processed_answerable.csv'),
        # ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_chatgpt_summary_generation_cxt_both/train_processed_sampled.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_chatgpt_summary_generation_cxt_both/train_processed_sampled.csv'),
        
        # ('data/modified/radqa/train_16_processed.json', 'data/modified/radqa/train_16_processed.csv'),
        
        # ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_chatgpt_summary_generation_cxt_both/0_16/train_processed.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_chatgpt_summary_generation_cxt_both/0_16/train_processed.csv'),
        # ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_chatgpt_summary_generation_cxt_both/0_16/train_processed_sampled.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_chatgpt_summary_generation_cxt_both/0_16/train_processed_sampled.csv'),
        ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_report_simple/0_32/train_processed.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_report_simple/0_32/train_processed.csv'),
        ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_report_simple/0_32/train_processed_sampled.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_report_simple/0_32/train_processed_sampled.csv'),
        
        # ('data/output/radqa/train/gpt-3.5-turbo-0613/answer_generation_question_generation_summary_chatgpt_summary_generation_cxt_both/0_16/train_processed.json', 'data/output/radqa/train/gpt-3.5-turbo-0613/answer_generation_question_generation_summary_chatgpt_summary_generation_cxt_both/0_16/train_processed.csv'),
        # ('data/output/radqa/train/gpt-3.5-turbo-0613/answer_generation_question_generation_summary_chatgpt_summary_generation_cxt_both/0_16/train_processed_sampled.json', 'data/output/radqa/train/gpt-3.5-turbo-0613/answer_generation_question_generation_summary_chatgpt_summary_generation_cxt_both/0_16/train_processed_sampled.csv'),
        
        # ('data/output/radqa/train/gpt-4o-2024-05-13/answer_generation_question_generation_summary_chatgpt_summary_generation_cxt_both/0_16/train_processed.json', 'data/output/radqa/train/gpt-4o-2024-05-13/answer_generation_question_generation_summary_chatgpt_summary_generation_cxt_both/0_16/train_processed.csv'),
        # ('data/output/radqa/train/gpt-4o-2024-05-13/answer_generation_question_generation_summary_chatgpt_summary_generation_cxt_both/0_16/train_processed_sampled.json', 'data/output/radqa/train/gpt-4o-2024-05-13/answer_generation_question_generation_summary_chatgpt_summary_generation_cxt_both/0_16/train_processed_sampled.csv'),
        # ('data/output/radqa/train/gpt-4o-2024-05-13/answer_generation_question_generation_report_simple/0_32/train_processed.json', 'data/output/radqa/train/gpt-4o-2024-05-13/answer_generation_question_generation_report_simple/0_32/train_processed.csv'),
        # ('data/output/radqa/train/gpt-4o-2024-05-13/answer_generation_question_generation_report_simple/0_32/train_processed_sampled.json', 'data/output/radqa/train/gpt-4o-2024-05-13/answer_generation_question_generation_report_simple/0_32/train_processed_sampled.csv'),
        
        # ('data/output/radqa/train/gpt-4o-2024-05-13/answer_generation_question_generation_summary_chatgpt_summary_generation_cxt_both/0_16/train_processed_unans_0.json', 'data/output/radqa/train/gpt-4o-2024-05-13/answer_generation_question_generation_summary_chatgpt_summary_generation_cxt_both/0_16/train_processed_unans_0.csv'),
        # ('data/output/radqa/train/gpt-4o-2024-05-13/answer_generation_question_generation_summary_chatgpt_summary_generation_cxt_both/0_16/train_processed_unans_32.json', 'data/output/radqa/train/gpt-4o-2024-05-13/answer_generation_question_generation_summary_chatgpt_summary_generation_cxt_both/0_16/train_processed_unans_32.csv'),
        # ('data/output/radqa/train/gpt-4o-2024-05-13/answer_generation_question_generation_summary_chatgpt_summary_generation_cxt_both/0_16/train_processed_unans_64.json', 'data/output/radqa/train/gpt-4o-2024-05-13/answer_generation_question_generation_summary_chatgpt_summary_generation_cxt_both/0_16/train_processed_unans_64.csv'),
        # ('data/output/radqa/train/gpt-4o-2024-05-13/answer_generation_question_generation_summary_chatgpt_summary_generation_cxt_both/0_16/train_processed_unans_96.json', 'data/output/radqa/train/gpt-4o-2024-05-13/answer_generation_question_generation_summary_chatgpt_summary_generation_cxt_both/0_16/train_processed_unans_96.csv'),
        # ('data/output/radqa/train/gpt-4o-2024-05-13/answer_generation_question_generation_summary_chatgpt_summary_generation_cxt_both/0_16/train_processed_unans_128.json', 'data/output/radqa/train/gpt-4o-2024-05-13/answer_generation_question_generation_summary_chatgpt_summary_generation_cxt_both/0_16/train_processed_unans_128.csv'),
        # ('data/output/radqa/train/gpt-4o-2024-05-13/answer_generation_question_generation_summary_chatgpt_summary_generation_cxt_both/0_16/train_processed_unans_160.json', 'data/output/radqa/train/gpt-4o-2024-05-13/answer_generation_question_generation_summary_chatgpt_summary_generation_cxt_both/0_16/train_processed_unans_160.csv'),
        
        # ('data/output/radqa/train/gpt-4o-2024-05-13/answer_generation_question_generation_report_simple/0_32/train_processed_unans_0.json', 'data/output/radqa/train/gpt-4o-2024-05-13/answer_generation_question_generation_report_simple/0_32/train_processed_unans_0.csv'),
        # ('data/output/radqa/train/gpt-4o-2024-05-13/answer_generation_question_generation_report_simple/0_32/train_processed_unans_32.json', 'data/output/radqa/train/gpt-4o-2024-05-13/answer_generation_question_generation_report_simple/0_32/train_processed_unans_32.csv'),
        # ('data/output/radqa/train/gpt-4o-2024-05-13/answer_generation_question_generation_report_simple/0_32/train_processed_unans_64.json', 'data/output/radqa/train/gpt-4o-2024-05-13/answer_generation_question_generation_report_simple/0_32/train_processed_unans_64.csv'),
        # ('data/output/radqa/train/gpt-4o-2024-05-13/answer_generation_question_generation_report_simple/0_32/train_processed_unans_96.json', 'data/output/radqa/train/gpt-4o-2024-05-13/answer_generation_question_generation_report_simple/0_32/train_processed_unans_96.csv'),
        # ('data/output/radqa/train/gpt-4o-2024-05-13/answer_generation_question_generation_report_simple/0_32/train_processed_unans_128.json', 'data/output/radqa/train/gpt-4o-2024-05-13/answer_generation_question_generation_report_simple/0_32/train_processed_unans_128.csv'),
        # ('data/output/radqa/train/gpt-4o-2024-05-13/answer_generation_question_generation_report_simple/0_32/train_processed_unans_160.json', 'data/output/radqa/train/gpt-4o-2024-05-13/answer_generation_question_generation_report_simple/0_32/train_processed_unans_160.csv'),
        
        ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_simple_summary_generation/0_32/train_processed.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_simple_summary_generation/0_32/train_processed.csv'),
        ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_simple_summary_generation/0_32/train_processed_sampled.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_simple_summary_generation/0_32/train_processed_sampled.csv'),
        
        # ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_simple_md_summary_generation_cxt_both_md/0_16/train_processed.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_simple_md_summary_generation_cxt_both_md/0_16/train_processed.csv'),
        # ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_simple_md_summary_generation_cxt_both_md/0_16/train_processed_sampled.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_simple_md_summary_generation_cxt_both_md/0_16/train_processed_sampled.csv'),
        
        # ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_report_simple_cxt_both/0_16/train_processed.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_report_simple_cxt_both/0_16/train_processed.csv'),
        # ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_report_simple_cxt_both/0_16/train_processed_sampled.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_report_simple_cxt_both/0_16/train_processed_sampled.csv'),
        
        # ('data/output/radqa/train/gpt-3.5-turbo-0613/answer_generation_question_generation_summary_simple_md_summary_generation_cxt_both_md/0_16/train_processed.json', 'data/output/radqa/train/gpt-3.5-turbo-0613/answer_generation_question_generation_summary_simple_md_summary_generation_cxt_both_md/0_16/train_processed.csv'),
        # ('data/output/radqa/train/gpt-3.5-turbo-0613/answer_generation_question_generation_summary_simple_md_summary_generation_cxt_both_md/0_16/train_processed_sampled.json', 'data/output/radqa/train/gpt-3.5-turbo-0613/answer_generation_question_generation_summary_simple_md_summary_generation_cxt_both_md/0_16/train_processed_sampled.csv'),
        
        # ('data/output/radqa/train/gpt-3.5-turbo-0613/answer_generation_question_generation_report_simple_cxt_both/0_16/train_processed.json', 'data/output/radqa/train/gpt-3.5-turbo-0613/answer_generation_question_generation_report_simple_cxt_both/0_16/train_processed.csv'),
        # ('data/output/radqa/train/gpt-3.5-turbo-0613/answer_generation_question_generation_report_simple_cxt_both/0_16/train_processed_sampled.json', 'data/output/radqa/train/gpt-3.5-turbo-0613/answer_generation_question_generation_report_simple_cxt_both/0_16/train_processed_sampled.csv'),
        
        # ('data/output/radqa/train/gpt-3.5-turbo-0613/answer_generation_question_generation_summary_complex_md_summary_generation_cxt_both_md/0_16/train_processed.json', 'data/output/radqa/train/gpt-3.5-turbo-0613/answer_generation_question_generation_summary_complex_md_summary_generation_cxt_both_md/0_16/train_processed.csv'),
        # ('data/output/radqa/train/gpt-3.5-turbo-0613/answer_generation_question_generation_summary_complex_md_summary_generation_cxt_both_md/0_16/train_processed_sampled.json', 'data/output/radqa/train/gpt-3.5-turbo-0613/answer_generation_question_generation_summary_complex_md_summary_generation_cxt_both_md/0_16/train_processed_sampled.csv'),
        
        # ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_report_simple/train_processed_sampled_32.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_report_simple/train_processed_sampled_32.csv'),
        # ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_simple_summary_generation/train_processed_sampled_32.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_simple_summary_generation/train_processed_sampled_32.csv'),
        
        # ('data/output/radqa/train/gpt-3.5-turbo-0613/answer_generation_question_generation_summary_simple_summary_generation/0_32/train_processed.json', 'data/output/radqa/train/gpt-3.5-turbo-0613/answer_generation_question_generation_summary_simple_summary_generation/0_32/train_processed.csv'),
        # ('data/output/radqa/train/gpt-3.5-turbo-0613/answer_generation_question_generation_summary_simple_summary_generation/0_32/train_processed_sampled.json', 'data/output/radqa/train/gpt-3.5-turbo-0613/answer_generation_question_generation_summary_simple_summary_generation/0_32/train_processed_sampled.csv'),
        
        # ('data/output/radqa/train/gpt-3.5-turbo-0613/answer_generation_question_generation_report_simple/0_32/train_processed.json', 'data/output/radqa/train/gpt-3.5-turbo-0613/answer_generation_question_generation_report_simple/0_32/train_processed.csv'),
        # ('data/output/radqa/train/gpt-3.5-turbo-0613/answer_generation_question_generation_report_simple/0_32/train_processed_sampled.json', 'data/output/radqa/train/gpt-3.5-turbo-0613/answer_generation_question_generation_report_simple/0_32/train_processed_sampled.csv'),
        
        # ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_simple_summary_generation_cxt_both/0_16/train_processed.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_simple_summary_generation_cxt_both/0_16/train_processed.csv'),
        # ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_simple_summary_generation_cxt_both/0_16/train_processed_sampled.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_summary_simple_summary_generation_cxt_both/0_16/train_processed_sampled.csv'),
        
        # ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_report_simple_cxt_both/0_16/train_processed.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_report_simple_cxt_both/0_16/train_processed.csv'),
        # ('data/output/radqa/train/llama3-8b/answer_generation_question_generation_report_simple_cxt_both/0_16/train_processed_sampled.json', 'data/output/radqa/train/llama3-8b/answer_generation_question_generation_report_simple_cxt_both/0_16/train_processed_sampled.csv'),
        
        ('data/output/radqa/train/gpt-3.5-turbo-0613/answer_generation_question_generation_summary_simple_summary_generation_cxt_both/0_16/train_processed.json', 'data/output/radqa/train/gpt-3.5-turbo-0613/answer_generation_question_generation_summary_simple_summary_generation_cxt_both/0_16/train_processed.csv'),
        ('data/output/radqa/train/gpt-3.5-turbo-0613/answer_generation_question_generation_summary_simple_summary_generation_cxt_both/0_16/train_processed_sampled.json', 'data/output/radqa/train/gpt-3.5-turbo-0613/answer_generation_question_generation_summary_simple_summary_generation_cxt_both/0_16/train_processed_sampled.csv'),
        
        ('data/output/radqa/train/gpt-3.5-turbo-0613/answer_generation_question_generation_report_simple_cxt_both/0_16/train_processed.json', 'data/output/radqa/train/gpt-3.5-turbo-0613/answer_generation_question_generation_report_simple_cxt_both/0_16/train_processed.csv'),
        ('data/output/radqa/train/gpt-3.5-turbo-0613/answer_generation_question_generation_report_simple_cxt_both/0_16/train_processed_sampled.json', 'data/output/radqa/train/gpt-3.5-turbo-0613/answer_generation_question_generation_report_simple_cxt_both/0_16/train_processed_sampled.csv'),
        
    ]         
    
    for p in pairs:
        jf = json.load(open(p[0]))
        convert_squad_to_csv_fmt(jf, p[1])
        
    # bash scripts/preprocess/convert_radqa_format.sh

    
