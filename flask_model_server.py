
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__)

model_path = "investment_predictor.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found.")

model = pickle.load(open(model_path, "rb"))

@app.route("/predict-investment", methods=["POST"])
def predict_investment():
    try:
        data = request.json

        required_fields = [
            "income", "assets", "expenses", "debts",
            "liabilities", "savings", "profit"
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        input_df = pd.DataFrame([data])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return jsonify({
            "can_invest": int(prediction),
            "confidence": round(float(probability), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)
