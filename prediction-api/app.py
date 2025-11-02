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
DEPLOYED_MODEL_PATH = os.getenv("DEPLOYED_MODEL_PATH", "models/deployed_model.pkl")
RF_CANDIDATE_PATH = os.getenv("RF_CANDIDATE_PATH", "models/candidate_model_rf.pkl")
XGB_CANDIDATE_PATH = os.getenv("XGB_CANDIDATE_PATH", "models/candidate_model_xgb.pkl")

# Expected features (must match your training data)
EXPECTED_FEATURES = ["magnitude", "cdi", "depth", "latitude", "longitude"]

# --- Globals for lazy init ---
_deployed_model = None
_rf_candidate = None
_xgb_candidate = None
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

def get_models():
    """Lazy-load models once, threadsafe"""
    global _deployed_model, _rf_candidate, _xgb_candidate
    
    if _deployed_model is None or _rf_candidate is None or _xgb_candidate is None:
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
            
            if _rf_candidate is None:
                try:
                    app.logger.info(f"Loading RF candidate from gs://{MODEL_BUCKET}/{RF_CANDIDATE_PATH}")
                    _rf_candidate = _load_model_from_gcs(MODEL_BUCKET, RF_CANDIDATE_PATH)
                    if _rf_candidate:
                        app.logger.info("✓ RF candidate loaded successfully")
                    else:
                        app.logger.warning("✗ RF candidate failed to load")
                except Exception as e:
                    app.logger.error(f"Failed to load RF candidate: {e}")
                    _rf_candidate = None
            
            if _xgb_candidate is None:
                try:
                    app.logger.info(f"Loading XGB candidate from gs://{MODEL_BUCKET}/{XGB_CANDIDATE_PATH}")
                    _xgb_candidate = _load_model_from_gcs(MODEL_BUCKET, XGB_CANDIDATE_PATH)
                    if _xgb_candidate:
                        app.logger.info("✓ XGB candidate loaded successfully")
                    else:
                        app.logger.warning("✗ XGB candidate failed to load")
                except Exception as e:
                    app.logger.error(f"Failed to load XGB candidate: {e}")
                    _xgb_candidate = None
    
    return _deployed_model, _rf_candidate, _xgb_candidate

@app.route("/healthz", methods=["GET"])
def health():
    deployed, rf_cand, xgb_cand = get_models()
    return jsonify({
        "status": "ok",
        "deployed_model": "loaded" if deployed else "failed",
        "rf_candidate": "loaded" if rf_cand else "failed",
        "xgb_candidate": "loaded" if xgb_cand else "failed",
        "expected_features": EXPECTED_FEATURES,
        "model_bucket": MODEL_BUCKET
    })

@app.route("/predict/", methods=["POST"])
def predict():
    """
    Predict tsunami risk using the deployed model.
    Falls back to candidate models if deployed model unavailable.
    
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
    deployed, rf_cand, xgb_cand = get_models()
    
    # Check if at least one model is loaded
    if deployed is None and rf_cand is None and xgb_cand is None:
        app.logger.error("No models available")
        return jsonify({
            "error": "Model not available",
            "message": "No models could be loaded from GCS",
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
        
        # Try deployed model first, fallback to ensemble of candidates
        model_used = None
        final_prediction = None
        final_probability = None
        
        # Priority 1: Use deployed model if available
        if deployed is not None:
            try:
                final_prediction = deployed.predict(df)[0]
                final_probability = deployed.predict_proba(df)[0][1] if hasattr(deployed, 'predict_proba') else final_prediction
                model_used = "deployed"
                app.logger.info(f"Using deployed model: prediction={final_prediction}, probability={final_probability}")
            except Exception as e:
                app.logger.warning(f"Deployed model prediction failed: {e}")
                model_used = None
        
        # Priority 2: Use ensemble of candidates if deployed model failed
        if model_used is None and (rf_cand is not None or xgb_cand is not None):
            predictions = []
            probabilities = []
            
            if rf_cand is not None:
                try:
                    rf_pred = rf_cand.predict(df)[0]
                    rf_proba = rf_cand.predict_proba(df)[0][1] if hasattr(rf_cand, 'predict_proba') else rf_pred
                    predictions.append(rf_pred)
                    probabilities.append(rf_proba)
                    app.logger.info(f"RF candidate: prediction={rf_pred}, probability={rf_proba}")
                except Exception as e:
                    app.logger.warning(f"RF candidate prediction failed: {e}")
            
            if xgb_cand is not None:
                try:
                    xgb_pred = xgb_cand.predict(df)[0]
                    xgb_proba = xgb_cand.predict_proba(df)[0][1] if hasattr(xgb_cand, 'predict_proba') else xgb_pred
                    predictions.append(xgb_pred)
                    probabilities.append(xgb_proba)
                    app.logger.info(f"XGB candidate: prediction={xgb_pred}, probability={xgb_proba}")
                except Exception as e:
                    app.logger.warning(f"XGB candidate prediction failed: {e}")
            
            if probabilities:
                final_probability = sum(probabilities) / len(probabilities)
                final_prediction = 1 if final_probability > 0.5 else 0
                model_used = "ensemble"
                app.logger.info(f"Using ensemble of candidates: prediction={final_prediction}, probability={final_probability}")
        
        # Generate response
        if final_prediction is not None and final_probability is not None:
            if final_prediction == 1:
                if final_probability > 0.7:
                    risk_level = "High"
                else:
                    risk_level = "Medium"
            else:
                risk_level = "Low"
            
            app.logger.info(f"Final prediction: {final_prediction}, risk: {risk_level}, model: {model_used}")
            
            return jsonify({
                "prediction": int(final_prediction),
                "tsunami_risk": risk_level,
                "probability": round(final_probability, 4),
                "message": "Tsunami likely" if final_prediction == 1 else "No tsunami expected",
                "features_used": EXPECTED_FEATURES,
                "model_used": model_used
            }), 200
        else:
            return jsonify({
                "error": "Prediction failed",
                "message": "All models failed to produce predictions",
                "prediction": None,
                "tsunami_risk": "Unknown"
            }), 500
        
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