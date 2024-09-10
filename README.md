Give me Some Hard Questions: Synthetic Data Generation for Clinical QA
====================================================================================================

<!-- # Method: InstrucTE
<img src="figures/method.png" > -->

# Installment

1. Create conda environment.
```
git clone https://github.com/bflashcp3f/synthetic-clinical-qa.git
cd synthetic-clinical-qa
conda env create -f environment.yaml
conda activate cqa
```

2. Set up OpenAI API key with the environment variable `OPENAI_API_KEY`. If you want to use Azure, set up the environment variable `AZURE_API_KEY`.

3. Install from the source
```
pip install -e .
```

# Generate Synthetic Data
We use RadQA as an example. For MIMIC-QA, just replace `radqa` with `mimicqa`.
```
# Direct Instruction (e.g., "generate 10 questions...")
bash scripts/generate/run_generate.sh radqa direct_instruction gpt-4o-2024-05-13

# Summarization + Direct Instruction
bash scripts/generate/run_generate.sh radqa summarization_direct_instruction gpt-4o-2024-05-13

# Summarization + No Overlap
bash scripts/generate/run_generate.sh radqa summarization_nonoverlap gpt-4o-2024-05-13

```

# Process Generated Data
Below is an example of selecting the top 5 questions from 10 generated questions of each document.
```
bash scripts/preprocess/process_llm_output_radqa.sh gpt-4o-2024-05-13 summarization_nonoverlap answer_generation_gpt4 10 5
```

# Supervised Fine-tuning
We used [BioClinRoBERTa](https://github.com/facebookresearch/bio-lm) (RoBERTa-large-PM-M3-Voc version) as the fine-tuning backbone. Make sure to download the model checkpoint and save it in `./models/BioClinRoBERTa/RoBERTa-large-PM-M3-Voc/`.

 Below is an example of fine-tuning on the generated data.
```
bash scripts/finetune/run_finetune_radqa.sh gpt-4o-2024-05-13 summarization_nonoverlap 0_64 generate_10_select_5 42
```