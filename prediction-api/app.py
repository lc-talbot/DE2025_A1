# app.py
import os, tempfile, threading
import joblib
from flask import Flask, request, jsonify
from google.cloud import storage

app = Flask(__name__)

# --- Config via env (with safe fallbacks) ---
MODEL_BUCKET   = os.getenv("_MODEL_BUCKET", "models_tsunami_2023019")
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "model_candidate.pkl")
MODEL_GCS_PATH = f"gs://{MODEL_BUCKET}/models/{MODEL_FILENAME}"

# --- Globals for lazy init ---
_model = None
_model_lock = threading.Lock()

def _load_model_from_gcs(gcs_path: str):
    client = storage.Client()
    bucket_name, blob_name = gcs_path.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    tmp = tempfile.NamedTemporaryFile(delete=False)
    blob.download_to_filename(tmp.name)
    return joblib.load(tmp.name)

def get_model():
    """Lazy-load the model once, threadsafe, Flask 3.x friendly."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:  # double-checked locking
                app.logger.info(f"Loading model from {MODEL_GCS_PATH} ...")
                _model = _load_model_from_gcs(MODEL_GCS_PATH)
                app.logger.info("Model loaded.")
    return _model

@app.route("/healthz", methods=["GET"])
def health():
    return {"status": "ok", "model": "loaded" if _model else "not_loaded"}

@app.route("/predict/", methods=["POST"])
def predict():
    # Ensure model is available (loads on first request)
    try:
        model = get_model()
    except Exception as e:
        app.logger.exception("Failed to load model")
        return jsonify({
            "error": "Model not available",
            "message": f"Model load failed: {e}",
            "prediction": None,
            "tsunami_risk": "Unknown"
        }), 503

    try:
        data = request.get_json(force=True, silent=False)
        # Expect a list of records
        import pandas as pd
        df = pd.DataFrame(data)

        y = model.predict(df)
        # Example: binary -> High/Low on first row
        pred = int(y[0])
        risk = "High" if pred == 1 else "Low"
        return jsonify({
            "prediction": pred,
            "tsunami_risk": risk,
            "message": "Model prediction successful"
        })
    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed", "message": str(e)}), 500
    




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Cloud Run injects PORT
    app.run(host="0.0.0.0", port=port, debug=True)