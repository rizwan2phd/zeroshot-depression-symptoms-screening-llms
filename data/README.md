# Data

## Dataset: ReDSM5 + Control Sentences

The experiments in this paper use **ReDSM5**, a corpus of  Reddit posts sentence-level annotated for DSM-5 Major Depressive Disorder symptoms by a licensed psychologist, extended with control sentences (labeled `NONE`).

> **Dataset homepage:** [https://huggingface.co/datasets/irlab-udc/redsm5](https://huggingface.co/datasets/irlab-udc/redsm5)  
> **License:** Apache 2.0  
> **Associated paper:** *ReDSM5: A Reddit Dataset for DSM-5 Depression Detection* (CIKM 2025) — [arXiv:2508.03399](https://arxiv.org/abs/2508.03399)

---

### Accessing the Dataset

ReDSM5 is a **gated dataset** — access requires accepting a data use agreement:

1. Download and complete the [ReDSM5 Agreement Form](https://www.irlab.org/ReDSM5_agreement.odt)
2. Submit it by email to [eliseo.bao@udc.es](mailto:eliseo.bao@udc.es)
3. Once approved, access the dataset at: [huggingface.co/datasets/irlab-udc/redsm5](https://huggingface.co/datasets/irlab-udc/redsm5)

A **free public sample** (25 paraphrased, anonymized entries) is available without any agreement at:  
[huggingface.co/datasets/irlab-udc/redsm5-sample](https://huggingface.co/datasets/irlab-udc/redsm5-sample)

---

### Dataset Files

| File | Description |
|---|---|
| `redsm5_annotations.csv` | Sentence-level annotations with DSM-5 symptom labels and clinical rationales |
| `redsm5_posts.csv` | Full Reddit post texts |

Sentences labeled `SPECIAL_CASE` are excluded from evaluation in this work.

---

### Symptom Distribution (status=1 only)

| Symptom | Posts |
|---|---|
| DEPRESSED_MOOD | 328 |
| WORTHLESSNESS | 311 |
| SUICIDAL_THOUGHTS | 165 |
| ANHEDONIA | 124 |
| FATIGUE | 124 |
| SLEEP_ISSUES | 102 |
| COGNITIVE_ISSUES | 59 |
| APPETITE_CHANGE | 44 |
| PSYCHOMOTOR | 35 |

---

### Experiment Input File (`dsm5_control_all.csv`)

The classification pipeline reads a file combining ReDSM5 annotated sentences with control sentences. This file is **not included** in the repository (it contains Reddit-derived text). Construct it from the ReDSM5 annotations after obtaining access.

**Expected columns:**

| Column | Description |
|---|---|
| `sentence_id` | Unique sentence identifier |
| `sentence_text` | The raw text to classify |
| `DSM5_symptom` | Ground truth DSM-5 symptom label |
| `status` | Annotation status (1 = positive evidence, 0 = explicit negative) |
| `<ModelName>` | Predicted label added by the classification pipeline |

**Example:**

```
sentence_id,sentence_text,DSM5_symptom,status
s_221_12_5,"I have trouble sleeping every night",SLEEP_ISSUES,1
s_001_ctrl,"The weather was nice today.",NONE,0
```

---

### Citing the Dataset

If you use ReDSM5, please also cite the original dataset paper:

```bibtex
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
