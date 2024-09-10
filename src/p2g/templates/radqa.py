
generate_qa_pairs_icl_llama2chat = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant.
<</SYS>>

Considering the provided context, create question-and-answer pairs. Ensure that the answers are directly derived from the specified context.

{{icl_examples}}

Context: {{input_context}}

Question:"""

generate_qa_pairs_complex_icl_llama2chat = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant.
<</SYS>>

Your task is to create precise and clinically relevant question-and-answer pairs from radiology reports, reflecting inquiries that a doctor or physician might have before ordering these exams. Review the provided report to formulate questions aimed at extracting specific, critical information. These questions should cover a broad spectrum of clinical interests, from the assessment of changes over time, such as the resolution of conditions or progression of diseases, to specific diagnostic concerns like obstructions, blood flow, and the presence of pathological findings. Questions should also assess the medical device placements and the effects of contrast media. The answers must be exact text spans (direct quotations) from the report, with no alterations, ensuring accuracy and direct relevance to the clinical context.

{{icl_examples}}

Context: {{input_context}}

Question:"""

run_qa_llama2chat = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant.
<</SYS>>

Please answer the question given the provided context. Ensure that the answer is a text span directly derived from the specified context.

Context: {{input_context}}

Question: {{input_question}}
Answer:"""

generate_question_llama2chat = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant.
<</SYS>>

Your task is to create precise and clinically questions given radiology reports and specified answers, reflecting inquiries that a doctor or physician might have before ordering these exams. Review the provided report to formulate questions aimed at extracting specific, critical information. These questions should cover a broad spectrum of clinical interests, from the assessment of changes over time, such as the resolution of conditions or progression of diseases, to specific diagnostic concerns like obstructions, blood flow, and the presence of pathological findings. Questions should also assess the medical device placements and the effects of contrast media.

Context: {{input_context}}

Answer: {{input_answer}}
Questions:
1."""

question_generation_report_simple = '''<radiology_report>
{{input_context}}
</radiology_report>

Considering the radiology report provided above, could you generate a list of questions from a medical professional's viewpoint that they would seek to address through a radiological examination (generate at most 15 questions formatted in an indexed list like "1. ... ", no separate categories needed)?'''

question_generation_report_simple_cxt_both = '''<radiology_report>
{{input_context}}
</radiology_report>

Considering the radiology report provided above, could you generate a list of questions from a medical professional's viewpoint that they would seek to address through a radiological examination (generate at most 15 questions formatted in an indexed list like "1. ... ", no separate categories needed)?'''

answer_generation = '''<radiology_report>
{{input_context}}
</radiology_report>

Please address the questions below by referencing the specific details provided in the preceding report. If a question cannot be answered with the information given, simply respond with "Unanswerable". For questions that can be answered, employ an extractive question-answering approach: provide only a quotation from the report as the answer, wrapped by quotation marks, and ensure these quotes are as concise as possible to accurately fulfill the query. The output should be formatted as "Q: ... <newline>A: ... <newline><newline>Q: ...".

{{input_questions}}'''

answer_generation_gpt4 = '''<radiology_report>
{{input_context}}
</radiology_report>

Please address the question below by referencing the specific details provided in the preceding report. Employ an extractive question-answering approach: provide only a quotation from the report as the answer, wrapped by quotation marks, and ensure these quotes are as concise as possible to accurately fulfill the query. Make every effort to find the answer in the report, considering all possible details. If, after thorough consideration, the question genuinely cannot be answered with the information provided, respond with "Unanswerable". Always aim to find a relevant and accurate answer. The output should be formatted as "Q: ... <newline>A: ... <newline><newline>Q: ...".

{{input_questions}}'''

qa_zs = '''<radiology_report>
{{input_context}}
</radiology_report>
  
Please answer the question below by referencing the specific details provided in the preceding report. If the question cannot be answered with the information given, simply respond with "Unanswerable". Otherwise, employ an extractive question-answering approach: provide only a quotation from the report as the answer, wrapped by quotation marks, and ensure these quotes are as concise as possible to accurately fulfill the query.
  
Q: {{input_question}}
A: '''

qa_zs_gpt4 = '''<radiology_report>
{{input_context}}
</radiology_report>
  
Please answer the question below by referencing the specific details provided in the preceding report. Employ an extractive question-answering approach: provide only a quotation from the report as the answer, wrapped by quotation marks, and ensure these quotes are as concise as possible to accurately fulfill the query. If the question cannot be answered with the information given, simply respond with "Unanswerable".

Q: {{input_question}}
A: '''

qa_zs_gpt4_ex = '''<radiology_report>
{{input_context}}
</radiology_report>
  
Please answer the question below by referencing the specific details provided in the preceding report. Employ an extractive question-answering approach: provide only a quotation from the report as the answer, wrapped by quotation marks, and ensure these quotes are as concise as possible to accurately fulfill the query. Please try to find the answer as hard as you can. If the question really cannot be answered with the information given, respond with "Unanswerable".

Q: {{input_question}}
A: '''

qa_zs_gpt4_ex2 = '''<radiology_report>
{{input_context}}
</radiology_report>
  
Please answer the question below by referencing the specific details provided in the preceding report. Employ an extractive question-answering approach: provide only a quotation from the report as the answer, wrapped by quotation marks, and ensure these quotes are as concise as possible to accurately fulfill the query. Make every effort to find the answer in the report, considering all possible details. If, after thorough consideration, the question genuinely cannot be answered with the information provided, respond with "Unanswerable". Always aim to find a relevant and accurate answer.

Q: {{input_question}}
A: '''

summary_generation_json = '''<radiology_report>
{{input_context}}
</radiology_report>

Output JSON Template:
{
  "symptoms": ["xx"],
  "possible_conditions": ["xx"],
  "medical_history": ["xx"],
  "examination_reasons": ["xx"],
  "areas_examined": ["xx"],
  "diagnostic_techniques": ["xx"]
}

Please summarize the radiology report above to cover 6 following aspects: "symptoms", "possible_conditions", "medical_history", "examination_reasons", "areas_examined", and "diagnostic_techniques" (e.g., CT scan), following the JSON template. If there is no information found for an aspect, then just output an empty list [] as the value in the JSON output.'''


summary_generation_cxt_both = '''<radiology_report>
{{input_context}}
</radiology_report>

Output JSON Template:
{
  "symptoms": ["xx"],
  "possible_conditions": ["xx"],
  "areas_examined": ["xx"],
  "patient_medical_history": ["xx"],
  "diagnostic_techniques": ["xx"]
}

Please summarize the radiology report above to cover 5 following aspects: "symptoms", "possible_conditions", "areas_examined", "patient_medical_history", and "diagnostic_techniques" (e.g., CT scan), following the JSON template. Please include the exact quotes from the report. If there is no information found for an aspect, then just output an empty list [] as the value in the JSON output.'''

summary_generation_cxt_both_md = '''**Task**: Generate a "clinical referral note" for a radiology report.

**Instructions**: You will be provided with a radiology report. Based on this information, create a "clinical referral note" that should include the following parts:

1. **Patient Information**: Personal information (e.g., age, gender) and relevant medical history.
2. **Symptoms and Medical Conditions**: The specific symptoms and medical conditions that led to the ordering of the radiology exam.
3. **Speculated Information for Verification**: Any related information speculated by the doctor, such as other possible symptoms, conditions, or findings that might not be explicitly stated in the report but can be verified during the radiology examination.

The "clinical referral note" should be concise and reflect the information needs of the ordering physician. If any part is not covered in the report, please indicate "None" for that section.

**Template for the Clinical Referral Note**:

---

**Clinical Referral Note**:

- **Patient Information**: [Provide patient's personal information and relevant medical history]
- **Symptoms and Medical Conditions**: [Describe the specific symptoms and medical conditions that led to the radiology exam]
- **Speculated Information for Verification**: [List the speculated information, such as other possible symptoms, conditions, or findings that can be verified during the radiology examination]

---

**Radiology Report**:
{{input_context}}

**Generated Clinical Referral Note**:
- **Patient Information**: 
- **Symptoms and Medical Conditions**: 
- **Speculated Information for Verification**: 

---
'''


question_generation_summary_simple_md = '''{{input_summary}}

Considering the clinical referral note provided, could you generate a list of questions from a medical professional's viewpoint that they would seek to address through a radiological examination (generate at most 15 questions formatted in an indexed list like "1. ... ", no separate categories needed)? If the clinical referral note has no useful information, just output "Empty input".'''

question_generation_summary_complex_md = '''{{input_summary}}

Considering the clinical referral note provided, could you generate a list of questions from a medical professional's viewpoint that they would seek to address through a radiological examination (generate at most 15 questions formatted in an indexed list like "1. ... ", no separate categories needed)? These questions should span a wide range of clinical interests. They should include evaluations of temporal changes, such as the improvement or progression of medical conditions, alongside specific diagnostic concerns, such as obstructions, vascular integrity, and the detection of any pathological anomalies. Additionally, the questions should encompass evaluations of medical device placements, if any, and the potential impacts of contrast agents used during the imaging process. Furthermore, these inquiries should consider hypothesizing about possible conditions and pertinent medical concepts that are implied but not explicitly mentioned in the patient's information. If the clinical referral note has no useful information, just output "Empty input".'''

question_generation_summary_complex = '''<patient_data>
{{input_summary}}
</patient_data>

Considering the patient data provided, could you generate a list of questions from a medical professional's viewpoint that they would seek to address through a radiological examination (generate at most 15 questions formatted in an indexed list, no separate categories needed)? These questions should span a wide range of clinical interests. They should include evaluations of temporal changes, such as the improvement or progression of medical conditions, alongside specific diagnostic concerns, such as obstructions, vascular integrity, and the detection of any pathological anomalies. Additionally, the questions should encompass evaluations of medical device placements, if any, and the potential impacts of contrast agents used during the imaging process. Furthermore, these inquiries should consider hypothesizing about possible conditions and pertinent medical concepts that are implied but not explicitly mentioned in the patient's information. If the patient data has no useful information, just output "Empty input".'''

question_generation_summary_chatgpt = '''<radiology_report_summary>
{{input_summary}}
</radiology_report_summary>

Considering the radiology report summary provided, create a set of questions that accurately reflect the information needs of physicians who order radiology exams (generate at most 15 questions formatted in an indexed list, no separate categories needed). These questions should includes the patient's possible medical condition and the reasons for the examination. When generating questions, consider both the explicit and implicit information needs conveyed in the summary. Aim for syntactical diversity by avoiding template-like questions. For example, instead of simply asking "Does the patient have CHF?", rephrase to capture implicit details such as "Did the cardiac silhouette enlarge due to CHF?" or "Is there any sign of pleural effusion?" If there is no useful information in the summary, just output "Empty input".'''

question_generation_plain = '''<radiology_report>
{{input_context}}
</radiology_report>

Considering the radiology report provided above, generate {{question_num}} questions from a medical professional's viewpoint that they would seek to address through a radiological examination, formatted in an indexed list like "1. ... ".'''

question_generation_nonoverlap = '''<radiology_report>
{{input_context}}
</radiology_report>

Considering the radiology report provided above, generate {{question_num}} questions from a medical professional's viewpoint that they would seek to address through a radiological examination, formatted in an indexed list like "1. ... ". Make sure that the generated questions do not contain any words from the radiology report.'''

question_generation_explicit = '''<radiology_report>
{{input_context}}
</radiology_report>

Considering the radiology report provided above, generate {{question_num}} questions from a medical professional's viewpoint that they would seek to address through a radiological examination. Please make sure each question starts with a different prefix, such as "is," "does," "has," "which," "what," "how," and "where", formatted in an indexed list like "1. ... ".'''

summary_generation_extra = '''<radiology_report>
{{input_context}}
</radiology_report>

Output JSON Template:
{
  "symptoms": ["xx"],
  "medical_conditions": ["xx"],
  "areas_examined": ["xx"],
  "patient_medical_history": ["xx"],
  "diagnostic_techniques": ["xx"],
  "other_potential_conditions": ["xx"],
}

Please generate a summary for the radiology report above to cover 6 following aspects: "symptoms", "medical_conditions", "areas_examined", "patient_medical_history", "diagnostic_techniques", and "other_potential_conditions" (relevant but unrevealed in the report), following the JSON template. If there is no information found for an aspect, then just output an empty list [] as the value in the JSON output.'''

summary_generation = '''<radiology_report>
{{input_context}}
</radiology_report>

Output JSON Template:
{
  "symptoms": ["xx"],
  "medical_conditions": ["xx"],
  "areas_examined": ["xx"],
  "patient_medical_history": ["xx"],
  "diagnostic_techniques": ["xx"],
}

Please generate a summary for the radiology report above to cover 5 following aspects: "symptoms", "medical_conditions", "areas_examined", "patient_medical_history", and "diagnostic_techniques", following the JSON template. If there is no information found for an aspect, then just output an empty list [] as the value in the JSON output.'''

# knowledge_elicitation = '''<patient_data>
# {{input_summary}}
# </patient_data>

# Considering the patient data provided, generate relevant but unspecified information for the following aspects: "potential_symptoms", "potential_medical_conditions", and "other_relevant_areas", following the JSON template below. If there is no information available for a particular aspect, just output an empty list [] as the value in the JSON output.

# Output JSON Template:
# {
#   "potential_symptoms": ["xx"],
#   "potential_medical_conditions": ["xx"],
#   "other_relevant_areas": ["xx"],
# }'''

knowledge_elicitation = '''<patient_data>
{{input_summary}}
</patient_data>

Considering the patient data provided, generate closely related but unmentioned knowledge for the following aspects: "symptoms", "medical_conditions", and "diagnostic_techniques". Follow the JSON template below, focusing on concepts that might interest medical professionals reviewing the patient data. If there is no information available for a particular aspect, just output an empty list [] as the value in the JSON output.

Output JSON Template:
{
  "symptoms": ["xx"],
  "medical_conditions": ["xx"],
  "diagnostic_techniques": ["xx"],
}'''

question_generation_summary = '''<patient_data>
{{input_summary}}
</patient_data>

Considering the patient data provided, generate {{question_num}} questions from a medical professional's viewpoint that they would seek to address through a radiological examination. Ensure the questions are diverse, covering various relevant aspects of the patient data. The generated questions should be formatted in an indexed list like "1. ... ".'''

# question_generation_summary_know = '''<patient_data>
# {{input_summary}}
# </patient_data>

# <external_knowledge>
# {{external_knowledge}}
# </external_knowledge>

# Considering the patient data and external knowledge provided, generate {{question_num}} questions from a medical professional's viewpoint that they would seek to address through a radiological examination. Prioritize incorporating external knowledge into the questions. Ensure the questions are diverse, covering various relevant aspects of the patient data and external knowledge. The generated questions should be formatted in an indexed list like "1. ... ".'''

question_generation_summary_know = '''<patient_data>
{{input_summary}}
</patient_data>

<external_knowledge>
{{external_knowledge}}
</external_knowledge>

Considering the patient data and external knowledge provided, generate {{question_num}} questions from a medical professional's viewpoint that they would seek to address through a radiological examination. Ensure the questions are diverse, covering various relevant aspects of the patient data. If the external knowledge is highly relevant, incorporate it into the questions. The generated questions should be formatted in an indexed list like "1. ... ".'''

question_generation_summary_know_only = '''<patient_data>
{{external_knowledge}}
</patient_data>

Considering the patient data provided, generate {{question_num}} questions from a medical professional's viewpoint that they would seek to address through a radiological examination. Ensure the questions are diverse, covering various relevant aspects of the patient data. The generated questions should be formatted in an indexed list like "1. ... ".'''

question_paraphrase = '''<radiology_report>
{{input_context}}
</radiology_report>

Q: {{input_question}}
A: {{input_answer}}

Please generate {{paraphrase_num}} paraphrased versions of the provided question. The paraphrased questions should:
- Paraphrase the question from short (question 1) to long (question {{paraphrase_num}}) while maintaining the original meaning and answer.
- Use different expressions for the same concept (e.g., full names to abbreviations and vice versa).
- Include various types of questions using different prefixes (e.g., "is," "does," "has," "which," "what," "how," "where," etc.), and allow for informal question formats, such as "Any signs of ...".
- Utilize different tenses where applicable.

Format the generated paraphrases in an indexed list like "1. ...".'''

question_paraphrase_nonoverlap = '''<radiology_report>
{{input_context}}
</radiology_report>

Q: {{input_question}}
A: {{input_answer}}

Please generate {{paraphrase_num}} paraphrased versions of the provided question. Ensure that the generated paraphrases do not use any words from the answer, by employing synonyms or abbreviations where necessary. If the question is unanswerable, rephrase it to avoid overlapping with the radiology report while maintaining relevance to the context. Format the generated paraphrases in an indexed list like "1. ...".'''

summary_generation_simple = '''<radiology_report>
{{input_context}}
</radiology_report>

Please generate a one-paragraph summary of the radiology report.'''

schema_generation = '''Output JSON Template:
{
  "key_attribute": ["attribute_1", "attribute_2", ..., "attribute_5"],
}

When a medical professional orders a radiology exam via a clinical note, what key high-level attributes should be considered to describe the patient's medical situation from the medical professional's perspective, prior to receiving the exam results? Please identify five attributes with concise names (using underscores) and present them in a list following the template above.'''

summary_generation_self_schema_gpt4 = '''<radiology_report>
{{input_context}}
</radiology_report>

Output JSON Template:
{
  "patient_history": ["xx"],
  "clinical_symptoms": ["xx"],
  "preliminary_diagnosis": ["xx"],
  "relevant_medications": ["xx"],
  "previous_imaging": ["xx"],
}

Please generate a summary for the radiology report above to cover 5 following aspects: "patient_history", "clinical_symptoms", "preliminary_diagnosis", "relevant_medications", and "relevant_medications", following the JSON template. If there is no information found for an aspect, then just output an empty list [] as the value in the JSON output.'''

summary_generation_self_schema_llama3 = '''<radiology_report>
{{input_context}}
</radiology_report>

Output JSON Template:
{
  "patient_condition": ["xx"],
  "exam_indication": ["xx"],
  "anatomical_site": ["xx"],
  "exam_type": ["xx"],
  "urgency_level": ["xx"],
}

Please generate a structured summary for the radiology report above to cover 5 following aspects: "patient_condition", "exam_indication", "anatomical_site", "exam_type", and "urgency_level", following the JSON template. If there is no information found for an aspect, then just output an empty list [] as the value in the JSON output.'''

question_generation_summary_nonoverlap = '''<patient_data>
{{input_summary}}
</patient_data>

Considering the patient data provided, generate {{question_num}} questions from a medical professional's viewpoint that they would seek to address through a radiological examination. Ensure the questions are diverse, covering various relevant aspects of the patient data. The generated questions should be formatted in an indexed list like "1. ... ". Make sure that the generated questions do not contain any words from the patient data.'''