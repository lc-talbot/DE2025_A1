# app.py
import os
import tempfile
import threading
import joblib
from flask import Flask, request, jsonify
from google.cloud import storage
import pandas as pd

app = Flask(__name__)

# --- Config via env (with safe fallbacks) ---
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "models_tsunami_2023019")
DEPLOYED_MODEL_PATH = os.getenv("DEPLOYED_MODEL_PATH", "model_candidate.pkl")

# Expected features (must match your training data)
EXPECTED_FEATURES = ["magnitude", "cdi", "depth", "latitude", "longitude"]

# --- Globals for lazy init ---
_deployed_model = None
_model_lock = threading.Lock()

def _load_model_from_gcs(bucket_name: str, blob_path: str):
    """Load a model from GCS path"""
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
        blob.download_to_filename(tmp.name)
        app.logger.info(f"Downloaded model from gs://{bucket_name}/{blob_path}")
        return joblib.load(tmp.name)
    except Exception as e:
        app.logger.error(f"Failed to load model from gs://{bucket_name}/{blob_path}: {e}")
        return None

def get_model():
    """Lazy-load deployed model once, threadsafe"""
    global _deployed_model
    
    if _deployed_model is None:
        with _model_lock:
            if _deployed_model is None:
                try:
                    app.logger.info(f"Loading deployed model from gs://{MODEL_BUCKET}/{DEPLOYED_MODEL_PATH}")
                    _deployed_model = _load_model_from_gcs(MODEL_BUCKET, DEPLOYED_MODEL_PATH)
                    if _deployed_model:
                        app.logger.info("✓ Deployed model loaded successfully")
                    else:
                        app.logger.warning("✗ Deployed model failed to load")
                except Exception as e:
                    app.logger.error(f"Failed to load deployed model: {e}")
                    _deployed_model = None
    
    return _deployed_model

@app.route("/healthz", methods=["GET"])
def health():
    model = get_model()
    return jsonify({
        "status": "ok",
        "deployed_model": "loaded" if model else "failed",
        "expected_features": EXPECTED_FEATURES,
        "model_bucket": MODEL_BUCKET
    })

@app.route("/predict/", methods=["POST"])
def predict():
    """
    Predict tsunami risk using the deployed model.
    
    Expected input (JSON list of records):
    [
      {
        "magnitude": 7.5,
        "cdi": 4,
        "depth": 25.0,
        "latitude": 38.2975,
        "longitude": 142.3729
      }
    ]
    """
    model = get_model()
    
    # Check if model is loaded
    if model is None:
        app.logger.error("Deployed model not available")
        return jsonify({
            "error": "Model not available",
            "message": "Deployed model could not be loaded from GCS",
            "prediction": None,
            "tsunami_risk": "Unknown"
        }), 503

    try:
        data = request.get_json(force=True, silent=False)
        app.logger.info(f"Prediction request received: {data}")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        app.logger.info(f"Input DataFrame shape: {df.shape}, columns: {list(df.columns)}")
        
        # Validate features
        missing_features = set(EXPECTED_FEATURES) - set(df.columns)
        if missing_features:
            return jsonify({
                "error": "Prediction failed",
                "message": f"Missing required features: {list(missing_features)}",
                "expected_features": EXPECTED_FEATURES,
                "received_features": list(df.columns),
                "prediction": None,
                "tsunami_risk": "Unknown"
            }), 400
        
        # Reorder columns to match training order
        df = df[EXPECTED_FEATURES]
        app.logger.info(f"Features reordered: {list(df.columns)}")
        
        # Use deployed model
        final_prediction = model.predict(df)[0]
        final_probability = model.predict_proba(df)[0][1] if hasattr(model, 'predict_proba') else final_prediction
        
        app.logger.info(f"Prediction: {final_prediction}, probability: {final_probability}")
        
        # Determine risk level
        if final_prediction == 1:
            if final_probability > 0.7:
                risk_level = "High"
            else:
                risk_level = "Medium"
        else:
            risk_level = "Low"
        
        app.logger.info(f"Final prediction: {final_prediction}, risk: {risk_level}")
        
        return jsonify({
            "prediction": int(final_prediction),
            "tsunami_risk": risk_level,
            "probability": round(final_probability, 4),
            "message": "Tsunami likely" if final_prediction == 1 else "No tsunami expected",
            "features_used": EXPECTED_FEATURES
        }), 200
        
    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed",
            "message": str(e),
            "prediction": None,
            "tsunami_risk": "Unknown"
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.logger.info(f"Starting API server on port {port}")
    app.logger.info(f"Model bucket: {MODEL_BUCKET}")
    app.logger.info(f"Expected features: {EXPECTED_FEATURES}")
    app.run(host="0.0.0.0", port=port, debug=False)