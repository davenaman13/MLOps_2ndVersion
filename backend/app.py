from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os
import logging

from utils import load_encoders, preprocess_input

app = Flask(__name__)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.FileHandler("backend.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("backend")
# ----------------------

base_dir = os.path.dirname(__file__)

with open(os.path.join(base_dir, 'model_weights.pkl'), 'rb') as f:
    model = pickle.load(f)

encoders = load_encoders(path=os.path.join(base_dir, 'encoders.pkl'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_data = request.json
        print("[BACKEND] Received data:", user_data)
        logger.info("Received data: %s", user_data)

        user_df = pd.DataFrame([user_data])
        print("[BACKEND] DataFrame:\n", user_df)
        logger.debug("DataFrame:\n%s", user_df)

        user_encoded = preprocess_input(user_df, encoders)
        print("[BACKEND] Encoded input:\n", user_encoded)
        logger.debug("Encoded input:\n%s", user_encoded)

        prediction = model.predict(user_encoded)[0]
        result = "Will Seek Treatment" if prediction == 1 else "Will Not Seek Treatment"
        print("[BACKEND] Prediction result:", result)
        logger.info("Prediction result: %s", result)
        return jsonify({'prediction': result})
    except Exception as e:
        print("[BACKEND] Error occurred:", str(e))
        logger.exception("Error during prediction")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/')
def home():
    return "🎉 Mental Health Risk Prediction API is running!"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)
