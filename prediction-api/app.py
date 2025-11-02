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
# These should point to where your NEW trained models are saved
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "data_tsunami_2023019")
RF_MODEL_PATH = os.getenv("RF_MODEL_PATH", "861179434993/tsunami-training-pipeline-latest/random-forest-trainer/model")
XGB_MODEL_PATH = os.getenv("XGB_MODEL_PATH", "861179434993/tsunami-training-pipeline-latest/xgboost-trainer/model")

# Expected features (must match your training data)
EXPECTED_FEATURES = ["magnitude", "cdi", "depth", "latitude", "longitude"]

# --- Globals for lazy init ---
_rf_model = None
_xgb_model = None
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
        raise

def get_models():
    """Lazy-load both models once, threadsafe"""
    global _rf_model, _xgb_model
    
    if _rf_model is None or _xgb_model is None:
        with _model_lock:
            if _rf_model is None or _xgb_model is None:
                try:
                    app.logger.info(f"Loading Random Forest model from gs://{MODEL_BUCKET}/{RF_MODEL_PATH}")
                    _rf_model = _load_model_from_gcs(MODEL_BUCKET, RF_MODEL_PATH)
                    app.logger.info("Random Forest model loaded successfully")
                except Exception as e:
                    app.logger.error(f"Failed to load RF model: {e}")
                    _rf_model = None
                
                try:
                    app.logger.info(f"Loading XGBoost model from gs://{MODEL_BUCKET}/{XGB_MODEL_PATH}")
                    _xgb_model = _load_model_from_gcs(MODEL_BUCKET, XGB_MODEL_PATH)
                    app.logger.info("XGBoost model loaded successfully")
                except Exception as e:
                    app.logger.error(f"Failed to load XGBoost model: {e}")
                    _xgb_model = None
    
    return _rf_model, _xgb_model

@app.route("/healthz", methods=["GET"])
def health():
    rf_model, xgb_model = get_models()
    return jsonify({
        "status": "ok",
        "random_forest": "loaded" if rf_model else "failed",
        "xgboost": "loaded" if xgb_model else "failed",
        "expected_features": EXPECTED_FEATURES
    })

@app.route("/predict/", methods=["POST"])
def predict():
    """
    Predict tsunami risk using ensemble of RF and XGBoost models.
    
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
    rf_model, xgb_model = get_models()
    
    # Check if at least one model is loaded
    if rf_model is None and xgb_model is None:
        app.logger.error("No models available")
        return jsonify({
            "error": "Model not available",
            "message": "Neither Random Forest nor XGBoost models could be loaded",
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
        
        # Make predictions with both models
        predictions = []
        probabilities = []
        
        if rf_model is not None:
            try:
                rf_pred = rf_model.predict(df)[0]
                rf_proba = rf_model.predict_proba(df)[0][1] if hasattr(rf_model, 'predict_proba') else rf_pred
                predictions.append(rf_pred)
                probabilities.append(rf_proba)
                app.logger.info(f"RF prediction: {rf_pred}, probability: {rf_proba}")
            except Exception as e:
                app.logger.warning(f"RF prediction failed: {e}")
        
        if xgb_model is not None:
            try:
                xgb_pred = xgb_model.predict(df)[0]
                xgb_proba = xgb_model.predict_proba(df)[0][1] if hasattr(xgb_model, 'predict_proba') else xgb_pred
                predictions.append(xgb_pred)
                probabilities.append(xgb_proba)
                app.logger.info(f"XGB prediction: {xgb_pred}, probability: {xgb_proba}")
            except Exception as e:
                app.logger.warning(f"XGB prediction failed: {e}")
        
        # Ensemble: average probabilities
        if probabilities:
            avg_probability = sum(probabilities) / len(probabilities)
            final_prediction = 1 if avg_probability > 0.5 else 0
            
            if final_prediction == 1:
                if avg_probability > 0.7:
                    risk_level = "High"
                else:
                    risk_level = "Medium"
            else:
                risk_level = "Low"
            
            app.logger.info(f"Ensemble prediction: {final_prediction}, probability: {avg_probability}, risk: {risk_level}")
            
            return jsonify({
                "prediction": int(final_prediction),
                "tsunami_risk": risk_level,
                "probability": round(avg_probability, 4),
                "message": "Tsunami likely" if final_prediction == 1 else "No tsunami expected",
                "features_used": EXPECTED_FEATURES
            }), 200
        else:
            return jsonify({
                "error": "Prediction failed",
                "message": "Both models failed to produce predictions",
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