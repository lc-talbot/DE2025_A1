import os
import tempfile
import joblib
from google.cloud import storage
from flask import Flask, request, jsonify

app = Flask(__name__)

# --- Configuration ---
MODEL_BUCKET = os.getenv("_MODEL_BUCKET", "your-default-model-bucket")  # fallback bucket
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "model_candidate.pkl")
MODEL_GCS_PATH = f"gs://{MODEL_BUCKET}/models/{MODEL_FILENAME}"

model = None  # global model variable

def load_model_from_gcs(gcs_path):
    """Download a model from GCS and load it with joblib."""
    try:
        client = storage.Client()
        bucket_name, blob_name = gcs_path.replace("gs://", "").split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        blob.download_to_filename(tmp_file.name)

        loaded_model = joblib.load(tmp_file.name)
        app.logger.info(f"✅ Loaded model from {gcs_path}")
        return loaded_model
    except Exception as e:
        app.logger.error(f"❌ Error loading model from {gcs_path}: {e}")
        return None


# --- Load model at startup ---
@app.before_first_request
def initialize_model():
    global model
    app.logger.info(f"Attempting to load model from {MODEL_GCS_PATH}")
    model = load_model_from_gcs(MODEL_GCS_PATH)
    if model is None:
        app.logger.warning("⚠️ Model not available; predictions will return error message.")


@app.route("/predict/", methods=["POST"])
def predict():
    global model
    data = request.get_json()
    if model is None:
        return jsonify({
            "error": "Model not available",
            "message": "Model is not loaded. Please check server logs.",
            "prediction": None,
            "tsunami_risk": "Unknown"
        }), 503

    try:
        import pandas as pd
        df = pd.DataFrame(data)
        prediction = model.predict(df)[0]
        risk = "High" if prediction == 1 else "Low"
        return jsonify({
            "prediction": int(prediction),
            "tsunami_risk": risk,
            "message": "Model prediction successful"
        })
    except Exception as e:
        return jsonify({"error": "Prediction failed", "message": str(e)}), 500