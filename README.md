<div align="center">

<img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
<img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" />
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
<img src="https://img.shields.io/badge/Status-Production--Ready-22c55e?style=for-the-badge" />

<br /><br />

# 🛡️ Credit Card Fraud Detection System

### A production-ready, cost-sensitive fraud detection pipeline combining machine learning with real-time decision logic — built to minimize financial loss, not just maximize accuracy.

<br />

[**Dataset**](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) · [**Architecture**](#-system-architecture) · [**Quick Start**](#-getting-started) · [**Roadmap**](#-future-roadmap)

<br />

```
ROC-AUC: ~0.98  ·  Fraud Recall: High  ·  Optimized Threshold: ~0.35
```

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [The Core Challenge](#-the-core-challenge)
- [System Architecture](#-system-architecture)
- [Approach & Techniques](#-approach--techniques)
- [Model Performance](#-model-performance)
- [Key Innovation: Cost-Based Threshold Optimization](#-key-innovation-cost-based-threshold-optimization)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Business Impact](#-business-impact)
- [Future Roadmap](#-future-roadmap)
- [Author](#-author)

---

## 🧭 Overview

Credit card fraud detection is not a pure classification problem — it is a **risk management system**.

This project goes far beyond a standard ML notebook. It implements a complete, end-to-end fraud detection pipeline designed to work in real-world conditions: from handling extreme class imbalance, to training a robust ensemble model, to deploying a **cost-aware decision engine** with a real-time Flask API and an interactive Streamlit dashboard.

---

## ⚠️ The Core Challenge

| Challenge | Detail |
|:---|:---|
| 🔢 **Class Imbalance** | Only ~0.17% of transactions are fraudulent |
| 📐 **High Dimensionality** | 28 anonymized PCA features (V1–V28) |
| 💸 **Asymmetric Cost** | Missing fraud is far more costly than a false alert |
| ⚡ **Real-Time Requirement** | Decisions must be made at transaction time |

A naïve classifier that predicts "not fraud" for everything achieves **99.83% accuracy** — while catching **zero fraud**. This project is engineered to solve that exact problem.

---

## 🏗️ System Architecture

```
                        ┌─────────────────────────┐
                        │    User Transaction      │
                        └────────────┬────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │       Flask API          │  ← Real-time inference endpoint
                        └────────────┬────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │     StandardScaler       │  ← Feature normalization
                        └────────────┬────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │   Random Forest Model    │  ← Fraud probability score
                        └────────────┬────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │  Cost-Based Threshold    │  ← Threshold: ~0.35
                        └────────────┬────────────┘
                                     │
                   ┌─────────────────┼─────────────────┐
                   ▼                 ▼                  ▼
           ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
           │  🟢 LOW RISK │ │ 🟡 MED RISK  │ │  🔴 HIGH RISK    │
           │   → Approve  │ │  → Review    │ │  → Block         │
           └──────────────┘ └──────────────┘ └──────────────────┘
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │   Streamlit Dashboard    │  ← Visualization & batch simulation
                        └─────────────────────────┘
```

---

## 🔬 Approach & Techniques

### 1. Handling Class Imbalance — SMOTE

The dataset is severely imbalanced (~0.17% fraud). Training on raw data produces a model that predicts "not fraud" almost always — technically accurate, but entirely useless for fraud detection.

**Solution: SMOTE (Synthetic Minority Over-sampling Technique)**

SMOTE generates synthetic fraud examples in feature space by interpolating between existing minority-class samples. This balances the training distribution without simply duplicating existing fraud records, enabling the model to learn meaningful, generalizable fraud patterns.

```
Before SMOTE:   [████████████████████████████████████░] 99.83% Legit | 0.17% Fraud
After SMOTE:    [████████████████████░░░░░░░░░░░░░░░░░] Balanced Training Set
```

---

### 2. Model — Random Forest Ensemble

A **Random Forest** was selected as the core classifier for the following reasons:

| Reason | Explanation |
|:---|:---|
| 🏋️ **Robustness** | Handles outliers and noisy features gracefully |
| 📊 **Feature Importance** | Native scoring for interpretability |
| 🎯 **Strong Baseline** | Excellent out-of-box performance on tabular data |
| 🛡️ **Anti-Overfitting** | Bagging reduces variance across trees |

The ensemble aggregates predictions from hundreds of decision trees, producing a calibrated **fraud probability score** (0.0 → 1.0) rather than a hard binary decision. This score is then passed to the cost-based threshold engine.

---

### 3. Feature Scaling

`StandardScaler` is applied to normalize `Amount` and `Time` features, which exist on entirely different scales compared to the anonymized PCA components V1–V28. This prevents those two raw features from disproportionately influencing model behavior.

---

## 📊 Model Performance

| Metric | Score |
|:---|:---|
| **ROC-AUC** | ~0.98 |
| **Recall (Fraud)** | High — captures the majority of fraud cases |
| **Precision** | Moderate — acceptable false positive rate given cost asymmetry |
| **Accuracy** | *(Deliberately de-emphasized — see note below)* |

> **Why not accuracy?**
>
> With only 0.17% fraud, a model that always predicts "safe" achieves **99.83% accuracy** while catching **zero fraud**. Accuracy is a misleading metric here. **ROC-AUC** (measures ranking quality across all thresholds) and **Recall** (measures how many actual frauds are caught) are the metrics that drive real business value.

---

## 💡 Key Innovation: Cost-Based Threshold Optimization

The single most impactful design decision in this system is **not** the choice of model — it's the **decision threshold**.

### The Problem with the Default Threshold (0.5)

Standard classifiers flag a transaction as fraudulent when its probability score exceeds **0.5**. This treats missed fraud and false alerts as **equally costly**. In the real world, they are anything but equal.

| Error Type | Consequence | Business Cost |
|:---|:---|:---:|
| ❌ **False Negative** (missed fraud) | Customer loses money · Bank absorbs loss · Trust permanently damaged | 🔴 Very High |
| ✅ **False Positive** (false alert) | Customer inconvenienced · Brief hold · Minor friction | 🟡 Low |

### The Solution: Lower the Threshold to ~0.35

By shifting the decision boundary from **0.5 → ~0.35**, the system becomes more aggressive at flagging suspicious activity. This deliberately trades some precision for significantly higher recall — which is exactly what real-world fraud prevention demands.

```
Fraud Probability Score
│
0.0 ──────────────── 0.35 ─────────────────────── 1.0
       [LOW RISK]     │        [MEDIUM / HIGH RISK]
                      ▲
              Optimal threshold
              (cost-minimizing)
```

**How the threshold was determined:**
The expected total cost was computed across every possible threshold value on the validation set. The threshold corresponding to the minimum total cost was selected — making this a **data-driven business decision**, not an arbitrary hyperparameter choice.

### Risk Tier Classification

| Probability Score | Risk Level | Action Taken |
|:---:|:---:|:---|
| `< 0.35` | 🟢 **LOW** | Transaction approved automatically |
| `0.35 – 0.65` | 🟡 **MEDIUM** | Flagged for human review |
| `> 0.65` | 🔴 **HIGH** | Transaction blocked immediately |

---

## ✨ Features

- **⚡ Real-time Prediction API** — Flask endpoint that accepts transaction features and returns a structured risk decision in milliseconds
- **📊 Interactive Dashboard** — Streamlit UI for exploring fraud probability distributions, threshold curves, and batch results visually
- **💰 Cost-Sensitive Decision Engine** — Classifies every transaction into LOW / MEDIUM / HIGH risk tiers based on cost-optimized thresholding
- **🎚️ Threshold Tuning Interface** — Experiment with different thresholds and instantly observe the cost/recall tradeoff on validation data
- **🔁 Batch Transaction Simulation** — Stress-test the model against simulated high-volume transaction streams

---

## 📁 Project Structure

```
Credit-Card-Fraud-Detection/
│
├── train.py            # Model training pipeline
│                       #   → SMOTE balancing
│                       #   → Random Forest training
│                       #   → Cost-based threshold optimization
│                       #   → Saves rf_model.pkl & scaler.pkl
│
├── app.py              # Flask API for real-time inference
│                       #   → POST /predict endpoint
│                       #   → Returns fraud_probability, risk_level, decision
│
├── dashboard.py        # Streamlit dashboard
│                       #   → Fraud probability distribution plots
│                       #   → Threshold sensitivity curves
│                       #   → Batch simulation interface
│
├── rf_model.pkl        # Trained Random Forest model (generated by train.py)
├── scaler.pkl          # Fitted StandardScaler (generated by train.py)
│
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip
- [Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — download `creditcard.csv` and place it in the project root

---

### Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 2 — Train the Model

```bash
python train.py
```

This will:
1. Load and preprocess `creditcard.csv`
2. Apply `StandardScaler` to `Amount` and `Time`
3. Apply **SMOTE** to balance the training set
4. Train a **Random Forest** classifier
5. Run **cost-based threshold optimization** on the validation set
6. Save `rf_model.pkl` and `scaler.pkl` to disk

---

### Step 3 — Start the Prediction API

```bash
python app.py
```

The Flask server starts at `http://localhost:5000`.

**Example Request:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, -1.3, 0.9, 0.4, -0.7, 1.2, ...]}'
```

**Example Response:**
```json
{
  "fraud_probability": 0.41,
  "risk_level": "HIGH",
  "decision": "BLOCK"
}
```

---

### Step 4 — Launch the Dashboard

```bash
streamlit run dashboard.py
```

Opens an interactive browser UI for threshold analysis, batch simulation, and result visualization.

---

## 💼 Business Impact

This system is designed to answer a real business question: **how do we minimize total financial loss from fraud, not just maximize model accuracy?**

| Outcome | How This System Delivers It |
|:---|:---|
| 💰 **Lower financial loss** | Cost-sensitive thresholding minimizes the total expected cost per decision |
| 🎯 **Higher fraud capture rate** | Optimized recall prioritizes catching fraud over avoiding false alerts |
| ⚡ **Operational speed** | Real-time API makes decisions at transaction time, not after the fact |
| 🔍 **Human-in-the-loop** | MEDIUM risk tier routes ambiguous cases to analysts — not everything is auto-blocked |
| 📈 **Adaptable** | Threshold can be re-tuned as fraud patterns, costs, or business priorities evolve |

---

## 🔭 Future Roadmap

- [ ] **SHAP Explainability** — identify which features drive individual fraud decisions; critical for regulatory compliance and analyst trust
- [ ] **Real-time Streaming** — integrate with Apache Kafka for high-throughput, low-latency transaction pipelines
- [ ] **Automated Retraining Pipeline** — detect model drift via statistical monitoring and trigger retraining with fresh labeled data
- [ ] **Cloud Deployment** — containerize with Docker and deploy to AWS / GCP / Render with auto-scaling
- [ ] **A/B Threshold Testing** — live comparison of threshold strategies across traffic segments to continuously optimize business outcomes
- [ ] **Graph-Based Fraud Detection** — model transaction networks to catch coordinated, multi-account fraud rings

---

## Screenshots



<img width="1919" height="220" alt="apitest" src="https://github.com/user-attachments/assets/60402b13-9691-4544-8fa8-002ef47f591e" />


<br /><br />


<img width="1920" height="1080" alt="dashboard" src="https://github.com/user-attachments/assets/8ff5263c-6b10-4287-94a7-f6ba7863a254" />

<br /><br />













<img width="1920" height="1080" alt="label" src="https://github.com/user-attachments/assets/258dfbd0-9085-4388-ba97-ffe89cbe3acf" />


<br /><br />











<img width="1920" height="1080" alt="predictionmanual" src="https://github.com/user-attachments/assets/30ca20f8-f29c-44b8-aea0-923d7507f6c2" />

<br /><br />

## 👤 Author

**Naef Nazar**

---

<div align="center">

If you found this useful, consider giving it a ⭐

</div>
