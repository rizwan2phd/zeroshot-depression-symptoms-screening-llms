# Resource-Efficient LLMs for Depression Symptoms Screening: Performance and Limitations in Zero-Shot Setting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-orange)](https://huggingface.co/)

Official code and data for the paper:

> **Resource-Efficient LLMs for Depression Symptoms Screening: Performance and Limitations in Zero-Shot Setting**  
> *Accepted at RaPID-6@MENTAL.ai@LREC2026*

---

## Overview

This repository contains the code for zero-shot classification of depression symptoms from text using multiple open-source large language models (LLMs). Models are evaluated on their ability to map free-text sentences to DSM-5 Major Depressive Disorder (MDD) symptom categories without any task-specific fine-tuning.

### DSM-5 Symptom Categories

| Category | Description |
|---|---|
| `DEPRESSED_MOOD` | Persistent sad, empty, or hopeless mood |
| `ANHEDONIA` | Loss of interest or pleasure in activities |
| `SLEEP_ISSUES` | Insomnia or hypersomnia |
| `FATIGUE` | Loss of energy or fatigue |
| `APPETITE_CHANGE` | Changes in appetite or weight |
| `COGNITIVE_ISSUES` | Difficulty concentrating or making decisions |
| `WORTHLESSNESS` | Feelings of worthlessness or excessive guilt |
| `PSYCHOMOTOR` | Psychomotor agitation or retardation |
| `SUICIDAL_THOUGHTS` | Recurrent thoughts of death or suicidal ideation |
| `NONE` | No depressive symptom present |

---

## Repository Structure

```
.
├── scripts/
│   └── category_pred.py        # Zero-shot LLM classification pipeline
├── notebooks/
│   └── dsm5_analysis.ipynb     # Results analysis and visualizations (Kaggle-ready)
├── data/
│   └── README.md               # Data description and access instructions
├── results/
│   └── README.md               # Output format description
├── requirements.txt            # Python dependencies
├── LICENSE
└── README.md
```

---

## Models Evaluated

The pipeline supports the following open-source LLMs (tested configurations):

| Model | Parameters | Family |
|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | 7B | Qwen2.5 |
| `Qwen/Qwen2.5-14B-Instruct` | 14B | Qwen2.5 |
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | Qwen2.5 |
| `Qwen/Qwen2.5-3B-Instruct` | 3B | Qwen2.5 |
| `meta-llama/Llama-3.2-1B-Instruct` | 1B | Llama 3.2 |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | Llama 3.2 |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | Mistral |
| `mistralai/Mistral-Nemo-Instruct-2407` | 12B | Mistral Nemo |

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/dsm5-llm-screening.git
cd dsm5-llm-screening
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. HuggingFace Authentication

Several models (e.g., Llama) require a HuggingFace account and acceptance of the model's license agreement.

```bash
export HUGGINGFACE_TOKEN=your_token_here
# or
huggingface-cli login
```

---

## Usage

### Zero-Shot Classification

```bash
python scripts/category_pred.py
```

By default, this runs classification with `mistralai/Mistral-Nemo-Instruct-2407`. Edit the `model_name` variable in the `__main__` block to switch models.

#### Input format

The script expects a CSV file (`dsm5_control_all.csv`) with at least a `sentence_text` column:

```
sentence_text
"I haven't been able to sleep for days."
"Everything feels pointless lately."
...
```

#### Output

Results are saved to `dsm5_control_all.csv` with a new column named after the model (e.g., `Mistral-Nemo-Instruct-2407`).

### Programmatic Usage

```python
from scripts.category_pred import TextClassifier

categories = [
    "DEPRESSED_MOOD", "WORTHLESSNESS", "ANHEDONIA",
    "SUICIDAL_THOUGHTS", "APPETITE_CHANGE", "SLEEP_ISSUES",
    "FATIGUE", "COGNITIVE_ISSUES", "PSYCHOMOTOR", "NONE"
]

domain_instruction = (
    "You are an expert clinical psychologist trained in DSM-5 diagnostic criteria "
    "for Major Depressive Disorder. Analyze the text carefully for indicators of "
    "depressive symptoms. If the text describes a depressive symptom, classify it "
    "into the one category. If the text shows NO depressive symptoms, classify it as NONE."
)

classifier = TextClassifier("Qwen/Qwen2.5-7B-Instruct")

result = classifier.classify(
    text="I can't find joy in things I used to love.",
    categories=categories,
    domain_instruction=domain_instruction
)
print(result)  # → ANHEDONIA
```

### Results Analysis

Open the Kaggle-compatible notebook for full evaluation:

```bash
jupyter notebook notebooks/dsm5_analysis.ipynb
```

The notebook computes:
- Per-symptom F1 scores (macro, weighted)
- Cohen's Kappa agreement
- False Negative Rate (FNR) and False Positive Rate (FPR) per symptom
- Misclassification analysis across all models

---

## Dataset

The experiments use **ReDSM5** ([irlab-udc/redsm5](https://huggingface.co/datasets/irlab-udc/redsm5)), a corpus of Reddit posts sentence-level annotated for DSM-5 MDD symptoms by a licensed psychologist (CIKM 2025). It is extended with control sentences labeled `NONE` to form the full evaluation set.

ReDSM5 is a **gated dataset** — to access it, complete the [data use agreement form](https://www.irlab.org/ReDSM5_agreement.odt) and email it to [eliseo.bao@udc.es](mailto:eliseo.bao@udc.es). A public anonymized sample of 25 entries is freely available at [irlab-udc/redsm5-sample](https://huggingface.co/datasets/irlab-udc/redsm5-sample).

> ⚠️ The dataset files are not included in this repository. See [`data/README.md`](data/README.md) for full access instructions and file format details.

---

## Citation

If you use this code or find our work useful, please cite both, our paper and the reference ReDSM5 dataset:

```bibtex
@inproceedings{rizwan2025dsm5llm,
  title     = {Resource-Efficient LLMs for Depression Symptoms Screening: Performance and Limitations in Zero-Shot Setting},
  author    = {Muhammad Rizwan and Jure Demšar},
  booktitle = {Proceedings of the RaPID-6 @LREC 2026},
  year      = {2026},
}

@misc{bao2025redsm5,
  title        = {ReDSM5: A Reddit Dataset for DSM-5 Depression Detection},
  author       = {Eliseo Bao and Anxo Pérez and Javier Parapar},
  year         = {2025},
  eprint       = {2508.03399},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2508.03399},
  note         = {Accepted at CIKM 2025}
}
```

## Acknowledgement


This publication has received funding from the European Union’s Horizon Europe research and innovation program under the Marie Sklodowska-Curie COFUND Postdoctoral Programme grant agreement No.101081355- SMASH and by the Republic of Slovenia and the European Union from the European Regional Development Fund.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

> **Ethical Note:** This work is intended for research purposes only. The classification system is not a clinical diagnostic tool and should not be used as a substitute for professional mental health assessment.
