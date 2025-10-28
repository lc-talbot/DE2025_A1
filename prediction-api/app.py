import os
from flask import Flask, request, jsonify
from earthquake_tsunami_predictor import TsunamiPredictor

# Create the Flask app
app = Flask(__name__)
app.config["DEBUG"] = os.environ.get('DEBUG', 'False') == 'True'

# Create an instance of your predictor class
tsunami_predictor = TsunamiPredictor()

@app.route('/', methods=['GET'])
def home():
    """
    Root endpoint - provides API information
    """
    return jsonify({
        "service": "Tsunami Prediction API",
        "version": "1.0",
        "endpoints": {
            "/predict/": "POST - Make tsunami predictions",
            "/health": "GET - Check service health",
            "/reload": "POST - Reload model from storage"
        },
        "status": "running"
    }), 200

@app.route('/predict/', methods=['POST'])
def predict():
    """
    Accepts a JSON input with earthquake/tsunami features
    and returns a risk prediction as JSON.
    
    Example request body:
    [
      {
        "Magnitude": 7.8,
        "Depth": 25.0,
        "Latitude": 38.5,
        "Longitude": 142.0
      }
    ]
    """
    try:
        prediction_input = request.get_json()
        
        # Validate input
        if not prediction_input:
            return jsonify({
                "error": "No input data provided",
                "message": "Please send JSON data in the request body"
            }), 400
        
        # Make prediction
        return tsunami_predictor.predict_single_record(prediction_input)
        
    except Exception as e:
        return jsonify({
            "error": "Invalid request",
            "message": str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint - returns service status and model availability
    """
    model_loaded = tsunami_predictor.model is not None
    
    health_info = {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_type": type(tsunami_predictor.model).__name__ if model_loaded else None,
        "model_bucket": tsunami_predictor.model_bucket,
        "model_file": tsunami_predictor.model_file
    }
    
    # Add feature information if available
    if model_loaded and tsunami_predictor.feature_names:
        health_info["expected_features"] = tsunami_predictor.feature_names
        health_info["num_features"] = len(tsunami_predictor.feature_names)
    
    return jsonify(health_info), 200 if model_loaded else 503

@app.route('/reload', methods=['POST'])
def reload():
    """
    Reload the model from Cloud Storage without restarting the service
    """
    try:
        success = tsunami_predictor.reload_model()
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Model reloaded successfully",
                "model_type": type(tsunami_predictor.model).__name__
            }), 200
        else:
            return jsonify({
                "status": "failed",
                "message": "Failed to reload model"
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Not found",
        "message": "The requested endpoint does not exist"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(port=port, host='0.0.0.0', debug=app.config["DEBUG"])