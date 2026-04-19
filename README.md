\# 🛡️ Credit Card Fraud Detection System


## Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud



> A production-ready, cost-sensitive fraud detection pipeline combining machine learning with real-time decision logic — built to minimize financial loss, not just maximize accuracy.



\---



\## 📌 Table of Contents



\- \[Overview](#overview)

\- \[The Core Challenge](#the-core-challenge)

\- \[System Architecture](#system-architecture)

\- \[Approach \& Techniques](#approach--techniques)

\- \[Model Performance](#model-performance)

\- \[Key Innovation: Cost-Based Threshold Optimization](#key-innovation-cost-based-threshold-optimization)

\- \[Features](#features)

\- \[Project Structure](#project-structure)

\- \[Getting Started](#getting-started)

\- \[Future Roadmap](#future-roadmap)

\- \[Author](#author)



\---



\## Overview



Credit card fraud detection is not a pure classification problem — it is a \*\*risk management system\*\*. This project goes beyond model accuracy to implement a full end-to-end fraud detection pipeline: from handling extreme class imbalance, to training a robust ensemble model, to deploying a cost-aware decision engine with a real-time Flask API and an interactive Streamlit dashboard.



\---



\## The Core Challenge



| Challenge | Detail |

|---|---|

| \*\*Class Imbalance\*\* | Only \~0.17% of transactions are fraudulent |

| \*\*High Dimensionality\*\* | 28 anonymized PCA features (V1–V28) |

| \*\*Asymmetric Cost\*\* | Missing fraud is far more costly than a false alert |

| \*\*Real-Time Requirement\*\* | Decisions must be made at transaction time |



\---



\## System Architecture



```

User Transaction

&#x20;     │

&#x20;     ▼

&#x20;Flask API  ──────────────────────── (Real-time inference endpoint)

&#x20;     │

&#x20;     ▼

StandardScaler  ──────────────────── (Feature normalization)

&#x20;     │

&#x20;     ▼

Random Forest Model  ─────────────── (Fraud probability score)

&#x20;     │

&#x20;     ▼

Cost-Based Threshold Logic  ──────── (Threshold: \~0.35)

&#x20;     │

&#x20;     ▼

Risk Decision

&#x20; ├── 🟢 LOW RISK     → Approve

&#x20; ├── 🟡 MEDIUM RISK  → Flag for review

&#x20; └── 🔴 HIGH RISK    → Block transaction

&#x20;     │

&#x20;     ▼

Streamlit Dashboard  ─────────────── (Visualization \& batch simulation)

```



\---



\## Approach \& Techniques



\### Handling Class Imbalance

The dataset is severely imbalanced (\~0.17% fraud). Training a naive classifier on raw data results in a model that predicts "not fraud" for nearly everything and still achieves 99.8% accuracy — which is useless.



\*\*Solution: SMOTE (Synthetic Minority Over-sampling Technique)\*\*  

Generates synthetic fraud examples in feature space to balance the training distribution, enabling the model to learn meaningful fraud patterns.



\### Model: Random Forest

A Random Forest ensemble was chosen for its:

\- Robustness to outliers and noisy features

\- Native feature importance scoring

\- Strong out-of-box performance on tabular data

\- Resistance to overfitting via bagging



\### Feature Scaling

`StandardScaler` is applied to normalize the `Amount` and `Time` features, which exist on very different scales compared to the anonymized PCA features V1–V28.



\---



\## Model Performance



| Metric | Score |

|---|---|

| \*\*ROC-AUC\*\* | \~0.98 |

| \*\*Recall (Fraud)\*\* | High — captures the majority of fraud cases |

| \*\*Precision\*\* | Moderate — acceptable false positive rate given cost asymmetry |



> \*\*Why not just use accuracy?\*\*  

> With 0.17% fraud, a model that predicts "safe" every time gets 99.83% accuracy but catches zero fraud. ROC-AUC and Recall are the metrics that matter here.



\---



\## Key Innovation: Cost-Based Threshold Optimization



The single most impactful design decision in this system is \*\*not\*\* the model — it's the decision threshold.



\### The Problem with Default Threshold (0.5)



Most classifiers output a probability score. By default, scores above 0.5 are flagged as fraud. But this treats false negatives (missed fraud) and false positives (false alerts) as equally costly. \*\*They are not.\*\*



| Error Type | Consequence | Relative Cost |

|---|---|---|

| \*\*False Negative\*\* (missed fraud) | Customer loses money, bank absorbs loss, trust damaged | 🔴 Very High |

| \*\*False Positive\*\* (false alert) | Customer call, minor friction, temporary hold | 🟡 Low |



\### The Solution: Lower the Threshold



By shifting the decision boundary from \*\*0.5 → \~0.35\*\*, the system becomes more aggressive at flagging suspicious transactions. This prioritizes recall (catching fraud) over precision (avoiding false alerts) — which aligns with real-world business priorities.



```

Fraud Probability Score

0.0 ──────────── 0.35 ──────────────────── 1.0

&#x20;    \[LOW RISK]  │  \[MEDIUM / HIGH RISK]

&#x20;                ▲

&#x20;         Optimal threshold

&#x20;         (cost-optimized)

```



This threshold was determined by computing the total expected cost across all thresholds on the validation set and selecting the minimum.



\---



\## Features



\- \*\*Real-time prediction API\*\* via Flask — accepts transaction data and returns a risk decision instantly

\- \*\*Interactive dashboard\*\* via Streamlit — visualize fraud probability distributions, threshold curves, and batch results

\- \*\*Cost-sensitive decision engine\*\* — classifies transactions into LOW / MEDIUM / HIGH risk tiers

\- \*\*Threshold tuning interface\*\* — experiment with different thresholds and observe cost/recall tradeoffs

\- \*\*Batch transaction simulation\*\* — stress-test the model against simulated transaction streams



\---



\## Project Structure



```

Credit-Card-Fraud-Detection/

│

├── train.py            # Model training pipeline (SMOTE, RF, threshold optimization)

├── app.py              # Flask API for real-time inference

├── dashboard.py        # Streamlit dashboard for visualization

│

├── rf\_model.pkl        # Trained Random Forest model

├── scaler.pkl          # Fitted StandardScaler

│

├── requirements.txt    # Python dependencies

└── README.md

```



\---



\## Getting Started



\### Prerequisites

\- Python 3.8+

\- pip



\### 1. Install Dependencies



```bash

pip install -r requirements.txt

```



\### 2. Train the Model



```bash

python train.py

```



This will preprocess the data, apply SMOTE, train the Random Forest, run cost-based threshold optimization, and save `rf\_model.pkl` and `scaler.pkl`.



\### 3. Start the Prediction API



```bash

python app.py

```



The Flask server will start at `http://localhost:5000`. Send a POST request with transaction features to receive a fraud risk decision.



\*\*Example request:\*\*

```bash

curl -X POST http://localhost:5000/predict \\

&#x20; -H "Content-Type: application/json" \\

&#x20; -d '{"features": \[0.1, -1.3, 0.9, ...]}'

```



\*\*Example response:\*\*

```json

{

&#x20; "fraud\_probability": 0.41,

&#x20; "risk\_level": "HIGH",

&#x20; "decision": "BLOCK"

}

```



\### 4. Launch the Dashboard



```bash

streamlit run dashboard.py

```



Opens an interactive UI in your browser for exploration, batch simulation, and threshold analysis.



\---



\## Future Roadmap



\- \[ ] \*\*SHAP Explainability\*\* — understand which features drive individual fraud decisions

\- \[ ] \*\*Real-time Streaming\*\* — integrate with Apache Kafka for high-throughput transaction pipelines

\- \[ ] \*\*Automated Retraining Pipeline\*\* — detect model drift and trigger retraining with fresh data

\- \[ ] \*\*Cloud Deployment\*\* — containerize with Docker and deploy to AWS / Render

\- \[ ] \*\*A/B Threshold Testing\*\* — live comparison of threshold strategies across traffic segments



\---



\## Author



\*\*Naef Nazar\*\*



> \*"Fraud detection is not just classification — it is a risk management system."\*

