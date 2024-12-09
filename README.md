# Give me Some Hard Questions: Synthetic Data Generation for Clinical QA

This repo provides code and data associated with ML4H 2024 Findings paper ["Give me Some Hard Questions: Synthetic Data Generation for Clinical QA"](https://arxiv.org/abs/2412.04573).
```
@misc{bai2024hardquestionssyntheticdata,
      title={Give me Some Hard Questions: Synthetic Data Generation for Clinical QA}, 
      author={Fan Bai and Keith Harrigian and Joel Stremmel and Hamid Hassanzadeh and Ardavan Saeedi and Mark Dredze},
      year={2024},
      eprint={2412.04573},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.04573}, 
}
```

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

# Data
We experiment with two Clinical QA datasets: [RadQA](https://physionet.org/content/radqa/1.0.0/) and [MIMIC-QA](https://physionet.org/content/mimic-iii-question-answer/1.0.0/), both of which can be accessed through PhysioNet.

# Synthetic Data Generation
## RadQA
```
# Direct Instruction (e.g., "generate 10 questions...")
bash scripts/generate/run_generate.sh radqa direct_instruction gpt-4o-2024-05-13

# Summarization + No Overlap
bash scripts/generate/run_generate.sh radqa summarization_nonoverlap gpt-4o-2024-05-13
```
## MIMIC-QA
```
# Direct Instruction
bash scripts/generate/run_generate.sh mimicqa direct_instruction gpt-4o-2024-05-13

# Summarization + Question Prefix
bash scripts/generate/run_generate.sh mimicqa summarization_explicit gpt-4o-2024-05-13
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