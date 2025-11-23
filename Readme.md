# Toxic Comment Classification Assessment
### Module: Neural Language Processing

## Project Overview
This project performs a comparative evaluation of Natural Language Processing (NLP) pipelines for the detection of abusive and toxic content in social media text.

The study aims to replicate and improve upon standard content moderation systems by comparing a statistical baseline against state-of-the-art Transformer architectures. A key focus of the research is addressing the **"Precision-Recall Trade-off"** in safety-critical applications, where identifying implicit threats (high recall) is prioritized over general accuracy.

Generative AI Declaration

Tools Used: ChatGPT / Gemini / GitHub Copilot

Purpose: Used for debugging code errors, generating boilerplate code for plotting charts, and refining the technical explanations in the report.

Modification: All AI-generated suggestions were reviewed, tested, and adapted to fit the specific dataset and project requirements. The final implementation and critical analysis are my own work.

##  Dataset
**Source:** [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) (Kaggle)
**Task:** Binary Classification (Safe vs. Abusive)
**Size:** 160k samples (Training), 64k samples (Test)

The original multi-label dataset (toxic, severe_toxic, obscene, threat, insult, identity_hate) was aggregated into a binary target `is_abusive` to focus on general malicious intent.

##  Methodology
The project implements and evaluates three distinct NLP pipelines:

### 1. Baseline Pipeline (Statistical)
* **Representation:** TF-IDF (Term Frequency - Inverse Document Frequency)
* **Algorithm:** Logistic Regression
* **Goal:** Establish a benchmark for computational efficiency and keyword-based detection.

### 2. Deep Learning Pipeline (Contextual)
* **Representation:** DistilBERT Tokenizer (Sub-word tokenization)
* **Algorithm:** DistilBERT (Fine-tuned Transformer)
* **Goal:** Evaluate the impact of contextual embeddings on detecting sarcasm and implicit toxicity.

### 3. Advanced Optimization (Novel Contribution)
* **Representation:** RoBERTa Tokenizer
* **Algorithm:** RoBERTa-Large with **Weighted Cross-Entropy Loss**
* **Hardware Acceleration:** Training performed on NVIDIA RTX 3090 (24GB VRAM)
* **Goal:** Maximize **Recall** to near-safety-critical levels (~90%) by leveraging a larger parameter space and penalizing false negatives.

##  Key Results
The evaluation demonstrates that while statistical methods achieve high accuracy, they fail to detect subtle abuse. The **RoBERTa-Large** model significantly outperformed the baseline in detecting threats.

| Pipeline | Model | Accuracy | F1-Score | Recall (Safety) |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | LogReg (TF-IDF) | 94.0% | 0.64 | 49.2% |
| **Deep Learning** | DistilBERT | 96.3% | 0.82 | 78.2% |
| **Advanced** | **RoBERTa-Large** | **96.0%** | **0.83** | **90.9%** |

*Note: The Advanced model achieved a Recall of 90.9%, meaning it successfully identified over 90% of toxic comments, compared to only 49% for the baseline.*

## 🛠️ Installation & Usage

### Prerequisites
* Python 3.10+
* NVIDIA GPU recommended (Required for Pipeline 3)

### 1. Setup Environment
```bash
# Clone repository
git clone [https://github.com/YOUR_USERNAME/nlp-toxic-comment-analysis.git]
cd nlp-toxic-comment-analysis

# Install dependencies
pip install -r requirements.txt