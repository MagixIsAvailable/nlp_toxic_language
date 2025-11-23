# Toxic Comment Classification Assessment
### Module: Neural Language Processing

## Project Overview
This project performs a comparative evaluation of Natural Language Processing (NLP) pipelines for the detection of abusive and toxic content in social media text.

The study aims to improve upon standard content moderation systems by comparing a statistical baseline against state-of-the-art Transformer architectures. A key focus is addressing the **"Precision-Recall Trade-off"** in safety-critical applications, prioritizing the detection of implicit threats (High Recall).

**Real-World Application:**
Beyond theoretical evaluation, this project implements the optimized model into a **live speech-to-visuals safety filter**. This demonstrates the model's capability to process real-time audio streams and visualize toxicity levels instantaneously using TouchDesigner.

Generative AI Declaration

Tools Used:  Gemini / GitHub Copilot

Purpose: Used for debugging code errors, generating boilerplate code for plotting charts, and refining the technical explanations in the report.

Generative AI was used to accelerate the development of the live_inference.py script, specifically for handling audio stream libraries (SpeechRecognition, pyaudio) and threading the OSC network connection. The core logic for model loading and inference integration was implemented by me to ensure compatibility with the trained models.

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


##  Key Results
The evaluation demonstrates that while statistical methods achieve high accuracy, they fail to detect subtle abuse. 

| Pipeline | Model | Accuracy | F1-Score | Recall (Safety) |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | LogReg (TF-IDF) | 94.0% | 0.64 | 49.2% |
| **Deep Learning** | DistilBERT | 96.3% | 0.82 | 78.2% |

## 🎨 Real-Time Application: The "Safety Filter"
To validate the model's performance in a production environment, I developed a **Real-Time Audio Visualizer** that maps semantic toxicity to visual chaos.

* **Architecture:**
    * **Input:** Live Microphone Audio (Speech-to-Text via Google API).
    * **Processing:** DistilBERT Inference (Python).
    * **Output:** OSC (Open Sound Control) data stream to TouchDesigner.
* **Visual Feedback Strategy:**
    * **🔵 Safe State (Score < 0.5):** Particles exhibit harmonic, laminar flow (Blue/Cyan).
    * **🔴 Toxic State (Score > 0.8):** Particles exhibit turbulent, chaotic explosion (Red/Orange).

### 🎥 Live Demo

##  Installation & Usage

### Prerequisites
* Python 3.10+
* NVIDIA GPU recommended (Required for Pipeline 3)

### 1. Setup Environment
```bash
# Clone repository
git clone https://github.com/MagixIsAvailable/nlp_toxic_language.git
cd nlp_toxic_language

# Install dependencies
pip install -r requirements.txt

