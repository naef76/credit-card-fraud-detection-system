# ============================================
# APP.PY — UPDATED WITH OPTIMAL THRESHOLD
# ============================================

from flask import Flask, request, jsonify
import joblib
import numpy as np
import datetime

app = Flask(__name__)

# Load model + scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------
# THRESHOLDS (from cost optimization)
# -------------------------------
HIGH_THRESHOLD = 0.35
MEDIUM_THRESHOLD = 0.2

# -------------------------------
# HOME
# -------------------------------
@app.route("/")
def home():
    return "Fraud Detection API Running"

# -------------------------------
# PREDICT
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json.get("features")

        # Validation
        if not data or len(data) != 30:
            return jsonify({"error": "Provide exactly 30 features"}), 400

        # Convert + scale
        data = np.array(data).reshape(1, -1)
        data_scaled = scaler.transform(data)

        # Predict probability
        prob = model.predict_proba(data_scaled)[0][1]

        # Decision logic (updated)
        if prob > HIGH_THRESHOLD:
            risk = "HIGH"
        elif prob > MEDIUM_THRESHOLD:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        # Logging
        log_entry = f"{datetime.datetime.now()} | prob={prob:.4f} | risk={risk}\n"
        with open("logs.txt", "a") as f:
            f.write(log_entry)

        return jsonify({
            "fraud_probability": float(prob),
            "risk": risk
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)