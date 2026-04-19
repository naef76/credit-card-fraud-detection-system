# ============================================
# TRAIN.PY — FULL PIPELINE + ADVANCED VISUALIZATION
# ============================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    precision_recall_curve
)

from imblearn.over_sampling import SMOTE

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# -------------------------------
# SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -------------------------------
# SCALING
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# SMOTE
# -------------------------------
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# -------------------------------
# MODEL
# -------------------------------
rf_model = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    n_jobs=-1,
    random_state=42
)

rf_model.fit(X_train, y_train)

# -------------------------------
# PREDICTION
# -------------------------------
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

# -------------------------------
# METRICS
# -------------------------------
print("\n--- RANDOM FOREST RESULTS ---")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# ============================================
# ADVANCED VISUALIZATION
# ============================================

# 1. Precision-Recall Curve
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_prob)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

# 2. Threshold vs Precision & Recall
thresholds = np.linspace(0, 1, 50)
precisions, recalls = [], []

for t in thresholds:
    y_pred_t = (y_prob > t).astype(int)
    precisions.append(precision_score(y_test, y_pred_t))
    recalls.append(recall_score(y_test, y_pred_t))

plt.figure()
plt.plot(thresholds, precisions, label="Precision")
plt.plot(thresholds, recalls, label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision & Recall vs Threshold")
plt.legend()
plt.show()

# 3. Confusion components vs threshold
tp_list, fp_list, fn_list = [], [], []

for t in thresholds:
    y_pred_t = (y_prob > t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
    tp_list.append(tp)
    fp_list.append(fp)
    fn_list.append(fn)

plt.figure()
plt.plot(thresholds, tp_list, label="TP")
plt.plot(thresholds, fp_list, label="FP")
plt.plot(thresholds, fn_list, label="FN")
plt.xlabel("Threshold")
plt.ylabel("Count")
plt.title("Confusion Matrix Components")
plt.legend()
plt.show()

# 4. COST CURVE (MOST IMPORTANT)
FN_cost = 10000
FP_cost = 100

costs = []

for t in thresholds:
    y_pred_t = (y_prob > t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
    loss = FN_cost * fn + FP_cost * fp
    costs.append(loss)

plt.figure()
plt.plot(thresholds, costs)
plt.xlabel("Threshold")
plt.ylabel("Total Cost")
plt.title("Cost vs Threshold")
plt.show()

best_t = thresholds[np.argmin(costs)]
print("\nBest Threshold (min cost):", best_t)

# 5. Probability distribution
plt.figure()
sns.histplot(y_prob[y_test == 0], label="Legit", kde=True)
sns.histplot(y_prob[y_test == 1], label="Fraud", kde=True)
plt.legend()
plt.title("Probability Distribution")
plt.show()

# -------------------------------
# SAVE MODEL
# -------------------------------
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel + Scaler saved.")


import joblib

joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Saved model and scaler")