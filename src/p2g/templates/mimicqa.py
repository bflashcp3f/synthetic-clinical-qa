# qa_zs_gpt4_ex2 = '''<clinical_record>
# {{input_context}}
# </clinical_record>
    
# Please answer the question below by referencing the specific details provided in the preceding clinical record. Employ an extractive question-answering approach: provide only a quotation from the record as the answer, wrapped by quotation marks. Make sure to consider different parts of the record, such as "chief complaint", "diagnosis", "treatment plan", etc.

# Q: {{input_question}}
# A: '''

qa_zs_mimicqa = '''<clinical_record>
{{input_context}}
</clinical_record>
    
Please answer the question below by referencing the specific details provided in the preceding clinical record. Employ an extractive question-answering approach: provide only a quotation from the record as the answer, wrapped by quotation marks. The answer should always be taken from the clinical record and can range from a few words to one or two sentences. For questions beginning with phrases like "does the patient have," "is the patient," etc., ensure the answer is a direct quote from the record rather than a simple yes or no. 

Question: {{input_question}}
Answer (direct quote): '''

question_generation_plain = '''<clinical_record>
{{input_context}}
</clinical_record>

Considering the clinical record provided above, generate {{question_num}} questions from a medical professional's viewpoint, formatted in an indexed list like "1. ... <newline>2. ...".'''

question_generation_nonoverlap = '''<clinical_record>
{{input_context}}
</clinical_record>

Considering the clinical record provided above, generate {{question_num}} questions from a medical professional's viewpoint, formatted in an indexed list like "1. ... <newline>2. ...". Make sure that the generated questions do not contain any words from the clinical record.'''

question_generation_summary = '''<patient_data>
{{input_summary}}
</patient_data>

Considering the patient data provided above, generate {{question_num}} questions from a medical professional's viewpoint. Ensure the questions are diverse, covering various relevant aspects of the patient data. The generated questions should be formatted in an indexed list like "1. ... ".'''

question_generation_summary_nonoverlap = '''<patient_data>
{{input_summary}}
</patient_data>

Considering the patient data provided above, generate {{question_num}} questions from a medical professional's viewpoint. Ensure the questions are diverse, covering various relevant aspects of the patient data. The generated questions should be formatted in an indexed list like "1. ... ". Make sure that the generated questions do not contain any words from the patient data.'''

question_generation_summary_short = '''<patient_data>
{{input_summary}}
</patient_data>

Considering the patient data provided above, generate {{question_num}} concise questions from a medical professional's viewpoint. The questions should be very straightforward and easy to understand, around ten words each, to enable doctors quickly assess the patient's past and current health condition. The questions should be formatted in an indexed list like "1. ... ".'''

question_generation_summary_short_explicit = '''<patient_data>
{{input_summary}}
</patient_data>

Considering the patient data provided above, generate {{question_num}} concise questions from a medical professional's viewpoint. The questions should be straightforward and easy to understand, helping doctors quickly assess the patient's past and current health condition. Please make sure questions start with different prefixes, such as "is," "does," "has," "which," "what," "how," and "where", formatted in an indexed list like "1. ... ".'''

question_generation_summary_explicit = '''<patient_data>
{{input_summary}}
</patient_data>

Considering the patient data provided above, generate {{question_num}} questions from a medical professional's viewpoint. Ensure the questions are diverse, covering various relevant aspects of the patient data. Please make sure questions start with different prefixes, such as "is," "does," "has," "which," "what," "how," and "where", formatted in an indexed list like "1. ... ".'''

question_generation = '''<clinical_record>
{{input_context}}
</clinical_record>

Considering the clinical record provided above, generate {{question_num}} questions from a medical professional's viewpoint.'''

question_generation_explicit = '''<clinical_record>
{{input_context}}
</clinical_record>

Considering the clinical record provided above, generate {{question_num}} questions from a medical professional's viewpoint. Please make sure each question starts with a different prefix, such as "is," "does," "has," "which," "what," "how," and "where", formatted in an indexed list like "1. ... ".'''

answer_generation = '''<clinical_record>
{{input_context}}
</clinical_record>

Please address the questions below by referencing the specific details provided in the preceding clinical record. Employ an extractive question-answering approach: provide only a quotation from the record as the answer, wrapped by quotation marks. The answer should always be taken from the clinical record and can range from a few words to one or two sentences. For questions beginning with phrases like "does the patient have," "is the patient," etc., ensure the answer is a direct quote from the record rather than a simple yes or no. If, after thorough consideration, the question genuinely cannot be answered with the information provided, respond with “Unanswerable”. The output should be formated as "Q: ... <newline>A: ... <newline><newline>Q: ...".

{{input_questions}}'''


answer_generation_short = '''<clinical_record>
{{input_context}}
</clinical_record>

Please address the questions below by referencing the specific details provided in the preceding clinical record. Employ an extractive question-answering approach: provide only a short quotation from the record as the answer, wrapped by quotation marks. If, after thorough consideration, the question genuinely cannot be answered with the information provided, respond with "Unanswerable." Otherwise, the answer should always be taken directly from the clinical record and should be a few words (usually less than ten). The output should be formated as "Q: ... <newline>A: ... <newline><newline>Q: ...".

{{input_questions}}'''


qa_pair_generation_plain = '''<clinical_record>
{{input_context}}
</clinical_record>

Based on the provided clinical record, generate {{question_num}} question-and-answer pairs from a medical professional's viewpoint. Ensure each answer is a direct quote from the record and can range from a few words to one full sentence. Format the pairs as follows:

Q1: [Your question here]
A1: [Direct quote from the record]

Q2: [Your question here]
A2: [Direct quote from the record]

...'''


schema_generation = '''Output JSON Template:
{
  "key_attribute": ["attribute_1", "attribute_2", ..., "attribute_5"],
}

Given a clinical record of a patient, what key high-level attributes should be considered to describe the patient's medical situation (no personal information) from the medical professional's perspective? Please identify five attributes with concise names (using underscores) and present them in JSON format.'''


# schema_generation = '''Output JSON Template:
# {
#   "key_attribute": ["attribute_1", "attribute_2", ..., "attribute_5"],
# }

# Given a clinical record of a patient, what key high-level attributes should be considered to describe the patient's situation (no personal information) from the medical professional's perspective? Please identify five attributes with concise names (using underscores) and present them in JSON format.'''


summary_generation = '''<clinical_record>
{{input_context}}
</clinical_record>

Output JSON Template:
{
    "patient_history": ["value1", "value2", ..., "value5"],
    "diagnosis": ["value1", "value2", ..., "value5"],
    "symptoms": ["value1", "value2", ..., "value5"],
    "medical_conditions": ["value1", "value2", ..., "value5"],
    "exam_results": ["value1", "value2", ..., "value5"],
}

Please generate a structured summary for the clinical record above to cover 5 following aspects: "patient_history", "diagnosis", "symptoms", "medical_conditions", and "exam_results", following the JSON template. Identify five values for each aspect at most. If there is no information found for an aspect, then just output an empty list [] as the value in the JSON output.'''


summary_generation_self_schema_gpt4 = '''<clinical_record>
{{input_context}}
</clinical_record>

Output JSON Template:
{
    "medical_history": ["value1", "value2", ..., "value5"],
    "current_symptoms": ["value1", "value2", ..., "value5"],
    "diagnosis": ["value1", "value2", ..., "value5"],
    "treatment_plan": ["value1", "value2", ..., "value5"],
    "medications": ["value1", "value2", ..., "value5"],
}

Please generate a structured summary for the clinical record above to cover 5 following aspects: "medical_history", "current_symptoms", "diagnosis", "treatment_plan", and "medications", following the JSON template. Identify five values for each aspect at most. If there is no information found for an aspect, then just output an empty list [] as the value in the JSON output.'''


summary_generation_self_schema_llama3 = '''<clinical_record>
{{input_context}}
</clinical_record>

Output JSON Template:
{
    "condition": ["value1", "value2", ..., "value5"],
    "symptoms": ["value1", "value2", ..., "value5"],
    "diagnosis": ["value1", "value2", ..., "value5"],
    "treatment": ["value1", "value2", ..., "value5"],
    "outcome": ["value1", "value2", ..., "value5"],
}

Please generate a structured summary for the clinical record above to cover 5 following aspects: "condition", "symptoms", "diagnosis", "treatment", and "outcome", following the JSON template. Identify five values for each aspect at most. If there is no information found for an aspect, then just output an empty list [] as the value in the JSON output.'''
