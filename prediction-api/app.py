import os
from flask import Flask, request, jsonify
from earthquake_tsunami_predictor import TsunamiPredictor

# Create the Flask app
app = Flask(__name__)
app.config["DEBUG"] = os.environ.get('DEBUG', 'False') == 'True'

# Create an instance of your predictor class
# This will automatically load the model from GCS bucket
tsunami_predictor = TsunamiPredictor()

@app.route('/')
def home():
    return "Tsunami Prediction API is running!"

@app.route('/health')
def health():
    model_loaded = tsunami_predictor.model is not None
    return jsonify({
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint - expects JSON like:
    [
        {
            "feature1": value1,
            "feature2": value2,
            ...
        }
    ]
    """
    try:
        data = request.json
        
        # Validate input
        if not data:
            return jsonify({
                "error": "No data provided",
                "message": "Please send JSON data in the request body"
            }), 400
        
        # Use the TsunamiPredictor to make prediction
        return tsunami_predictor.predict_single_record(data)
        
    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "message": str(e)
        }), 500

@app.route('/reload-model', methods=['POST'])
def reload_model():
    """
    Endpoint to reload the model from GCS without restarting the service
    """
    try:
        success = tsunami_predictor.reload_model()
        if success:
            return jsonify({
                "status": "success",
                "message": "Model reloaded successfully"
            }), 200
        else:
            return jsonify({
                "status": "error",
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
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting Tsunami Prediction API on port {port}")
    print(f"Model bucket: {os.environ.get('MODEL_BUCKET', 'models_tsunami_2023019')}")
    print(f"Model file: {os.environ.get('MODEL_FILE', 'deployed_model.pkl')}")
    app.run(host='0.0.0.0', port=port, debug=False)