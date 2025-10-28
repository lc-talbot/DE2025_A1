import os
from flask import Flask, request, jsonify
from earthquake_tsunami_predictor import TsunamiPredictor

# Create the Flask app
app = Flask(__name__)
app.config["DEBUG"] = os.environ.get('DEBUG', 'False') == 'True'

# Create an instance of your predictor class
tsunami_predictor = TsunamiPredictor()

@app.route('/')
def home():
    return "Tsunami Prediction API is running!"

@app.route('/health')
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    # Your prediction logic here
    data = request.json
    # result = predict(data)
    return jsonify({"prediction": "result here"})

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
    app.run(host='0.0.0.0', port=port, debug=False)