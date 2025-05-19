from flask import Flask, render_template, request
import requests
import logging

app = Flask(__name__)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[
        logging.FileHandler("frontend.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("frontend")
# ----------------------

BACKEND_URL = "http://mental-health-backend-service:5001/predict"

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        form_data = request.form.to_dict()
        try:
            response = requests.post(BACKEND_URL, json=form_data)
            prediction = response.json().get("prediction")
            print("[FRONTEND] Got prediction:", prediction)
            logger.info("Got prediction: %s", prediction)
        except Exception as e:
            print("[FRONTEND] Error:", str(e))
            logger.exception("Error while contacting backend")
            prediction = f"Error: {e}"
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)
