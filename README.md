# 🛡️ AI-Powered Behavioral Anomaly Detection for Cybersecurity

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Honeywell Hackathon Submission**
> 
> A robust, hybrid Machine Learning pipeline designed to solve the two biggest problems in modern Security Operations Centers (SOCs): **Alert Fatigue** and **Lack of Context**.

## 📖 Overview

Traditional security rules fail to catch subtle, sequential attacks (like low-and-slow data exfiltration or insider drift) and rely on rigid static thresholds that drown analysts in false positives. 

This project introduces a **two-stage hybrid AI architecture**:
1. **Unsupervised Sequence Modeling**: A PyTorch GRU Autoencoder trained *exclusively* on benign access logs to learn normal behavioral bounds. It detects temporal anomalies via high reconstruction error, gracefully handling extreme class imbalance.
2. **Rule-Based Explainability Layer**: The Top 1% most anomalous events are passed through an interpretable Decision Tree classifier, which translates the raw feature deviations into plain-English explanations for analysts (e.g., *"Flagged as Lateral Movement: Accessed 3 new unusual resources in the last 10 sessions"*).

## ✨ Key Features

- **Synthetic Threat Generation**: A robust data generator simulating normal behavior alongside 7 specific MITRE ATT&CK patterns (Credential Stuffing, Impossible Travel, Brute Force, Lateral Movement, etc.).
- **Temporal Feature Engineering**: Transforms raw logs into 16 interpretable features (geo-velocity, rolling auth failures, time-based z-scores).
- **Cold-Start Resilience**: Gracefully falls back to population-level statistics for brand-new entities before they establish a baseline.
- **Concept Drift Adaptation**: Incrementally fine-tunes the sequence model on recent, benign-classified traffic to prevent behavioral profiles from decaying.
- **Interactive Analyst Dashboard**: A localized Streamlit UI providing a ranked alert queue, threat summaries, and rich contextual entity timelines.

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.9+ installed.

### Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/im-Johaan/hackathon.git
cd hackathon
pip install torch pandas scikit-learn streamlit plotly
```

### Running the Dashboard
To view the results and interact with the AI-scored alerts:
```bash
streamlit run app.py
```

### Running the Full Pipeline
If you wish to regenerate the data and retrain the models from scratch, execute the pipeline in this exact order:
1. `python data_generator.py` (Generates raw synthetic access logs)
2. `python feature_engineering.py` (Extracts sequence features)
3. `python anomaly_detection.py` (Trains the GRU Autoencoder and scores the dataset)
4. `python explainability_layer.py` (Classifies the Top 1% anomalies and generates explanations)

## 📊 Evaluation & Metrics

Our hybrid model was evaluated on a heavily imbalanced dataset (97% normal, 3% attacks):

- **Precision at Top 1% Alert Budget**: **98.0%**
- **Recall at Top 1% Alert Budget**: **18.0%** (Identified the most critical sequence deviations with near-perfect precision)
- **Credential Stuffing Detection**: 100% Precision / 100% Recall (112/112 caught in top 1%)
- **Lateral Movement Detection**: 86% Precision / 100% Recall (6/6 caught in top 1%)

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `data_generator.py` | Generates the synthetic access logs and injects attack patterns. |
| `feature_engineering.py` | Computes sliding windows, rolling counts, and geo-velocity metrics. |
| `anomaly_detection.py` | PyTorch GRU Autoencoder training, evaluation, and anomaly scoring. |
| `explainability_layer.py` | Decision Tree classifier applied to the top 1% alerts. |
| `app.py` | Streamlit analyst dashboard. |
| `hackathon_report.md` | Comprehensive methodology, evaluation, and future work report. |
| `feature_dictionary.md` | Detailed breakdown of all 16 engineered features. |
| `anomaly_detector_gru.pt` | Saved weights for the trained GRU Autoencoder. |

## 🛠️ Built With
- **PyTorch** (Deep Learning & Sequence Modeling)
- **Scikit-Learn** (Decision Trees, Metrics, Preprocessing)
- **Pandas & NumPy** (Data Manipulation)
- **Streamlit & Plotly** (Interactive UI & Data Visualization)
