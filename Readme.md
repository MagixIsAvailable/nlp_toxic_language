# Toxic Comment Classification Assessment
### Module: Neural Language Processing

##  Project Overview
This project performs a comparative evaluation of Natural Language Processing (NLP) pipelines for the detection of abusive and toxic content in social media text.

The study aims to improve upon standard content moderation systems by comparing a statistical baseline against state-of-the-art Transformer architectures. A key focus is addressing the **"Precision-Recall Trade-off"** in safety-critical applications, prioritizing the detection of implicit threats (High Recall).

**Real-World Application:**
Beyond theoretical evaluation, this project implements the optimized model into a **live speech-to-visuals safety filter**. This demonstrates the model's capability to process real-time audio streams and visualize toxicity levels instantaneously using [TouchDesigner](https://derivative.ca/).

## ⚠️ Generative AI Declaration
* **Tools Used:** Gemini / GitHub Copilot
* **Purpose:** Used for debugging code errors, generating boilerplate code for plotting charts, and refining the technical explanations in the report.
* **Specific Implementation:**
    * **Real-Time Filter (`live_inference.py`):** Co-developed with AI assistance to handle low-level audio drivers and threading.
    * **AI Data Audit (`auto_audit.py`):** The **"Teacher-Student" audit architecture was conceptually designed by me** to solve the data validation bottleneck. Generative AI was utilized to **generate the Python API wrappers** for the Google Gemini SDK. The logic for the audit loop, error flagging, and the integration into the project workflow remains my own work.
* **Modification:** All AI-generated suggestions were reviewed, tested, and adapted to fit the specific dataset and project requirements. The final implementation and critical analysis are my own work.

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

## 📊 Key Results
The evaluation demonstrates that while statistical methods achieve high accuracy, they fail to detect subtle abuse. 

| Pipeline | Model | Accuracy | F1-Score | Recall (Safety) |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | LogReg (TF-IDF) | 94.0% | 0.64 | 49.2% |
| **Deep Learning** | DistilBERT | 96.3% | 0.82 | 78.2% |

##  Real-Time Application: The "Safety Filter"
To validate the model's performance in a production environment, I developed a **Real-Time Audio Visualizer** that maps semantic toxicity to visual chaos.
### 🎥 Live Demo
[**▶️ Watch the Real-Time Safety Filter in Action**](https://youtu.be/QQauk5tuYt4)

> *Demonstration of the system reacting to live speech. Note the transition from Blue (Safe) to Red (Toxic) as the intent of the speech changes.*


* **Architecture:**
    * **Input:** Live Microphone Audio (Speech-to-Text via Google API).
    * **Processing:** DistilBERT Inference (Python).
    * **Output:** OSC (Open Sound Control) data stream to TouchDesigner.
* **Visual Feedback Strategy:**
    * **🔵 Safe State (Score < 0.5):** Particles exhibit harmonic, laminar flow (Blue/Cyan).
    * **🔴 Toxic State (Score > 0.8):** Particles exhibit turbulent, chaotic explosion (Red/Orange).
* ** Automated Data Annotation (Active Learning):** The system implements a "Data Flywheel." Inference results are automatically annotated (0=Safe, 1=Toxic) based on a confidence threshold (>0.5) and logged to `Data/live_recording_data.csv`. This creates a **pseudo-labeled dataset** that allows for future Human-in-the-Loop (HITL) fine-tuning, where an administrator only needs to correct false positives (e.g., "I hate broccoli") to retrain the model.

## 📂 Project Structure
```text
nlp_toxic_language/
├── Data/                   
│   ├── train.csv           # Training dataset (Download from Kaggle)
│   └── live_recording_data.csv # Generated logs from live inference
├── results/                # Trained DistilBERT model checkpoints (Excluded from repo)
├── TD/                     # TouchDesigner project files
│   └── toxic_laguage.1.toe
│   └──toxic_laguage.toe
├── nlp_toxic.ipynb         # Main Analysis & Training Notebook
├── live_inference.py       # Real-time Speech-to-Text & Inference Script
├── check_mic.py            # Utility script to find Microphone Index
├── requirements.txt        # Python library dependencies
├── .gitignore              # Files excluded from version control
├── README.md               # Project documentation
└── test_phrases.txt        #  Scripts to test the Safety Filter (Read these aloud)
````

##  System Requirements
This project was developed and tested in the following environment. While the training notebook can run on standard hardware (Google Colab / CPU), the **Real-Time Safety Filter** requires GPU acceleration for low-latency performance.

* **Development Hardware:**
    * **GPU:** NVIDIA GeForce RTX 3090 (24GB VRAM)
    * **CPU:** AMD Ryzen 9 5950X
    * **RAM:** 64GB
* **Software Environment:**
    * **OS:** Windows 11
    * **Python:** 3.12.6
    * **CUDA:** 12.1 (PyTorch optimized)

## 🔮 Future Work: AI-Assisted Data Audit
**Status:** *Proof of Concept / Experimental*

To address the limitations of the lightweight DistilBERT model, I conceptualized a **"Teacher-Student" validation loop**. This component (`auto_audit.py`) utilizes a Large Language Model (**Google Gemini Pro**) to act as an objective auditor for the live inference logs.

* **Concept:** Use a frozen, high-capacity LLM (Teacher) to validate and correct the predictions of the smaller, fine-tuned model (Student).
* **Workflow:**
    1.  **Ingest:** Reads `live_recording_data.csv`.
    2.  **Audit:** Sends each phrase to the Gemini API for a "Safety Verdict."
    3.  **Correction:** Compares Gemini's label against DistilBERT's label. Discrepancies (e.g., DistilBERT flagging "I hate broccoli") are flagged for retraining.
* **Goal:** To automate the creation of a "Gold Standard" dataset for continuous fine-tuning without manual human labelling.

## 🛠️ Installation & Usage

### Prerequisites
* Python 3.10+
* NVIDIA GPU recommended (Required for fast inference, tested on rtx3090)

### 1. Setup Environment
```bash
# Clone repository
git clone https://github.com/MagixIsAvailable/nlp_toxic_language.git
cd nlp_toxic_language

# Install dependencies
pip install -r requirements.txt
```

### 2\. Download Data

1.  Download `train.csv.zip` from [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data).
2.  Extract `train.csv` into a folder named `Data/` in the root directory.

### 3\. Run the Analysis

Open the Jupyter Notebook to view the training, evaluation, and visualization code.

```bash
jupyter notebook Toxic_Language_Evaluation.ipynb
```

### 4.  Run Live Inference
To initiate the real-time safety filter, ensure your microphone is connected.

1.  Run the script:
    ```bash
    python live_inference.py
    ```
2.  **Validation:** Open **`test_phrases.txt`** and read the sentences aloud.
    * The script will log the results to `Data/live_recording_data.csv`.
    * If TouchDesigner is open, the visualizer will react accordingly.