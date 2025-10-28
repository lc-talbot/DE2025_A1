import os
from flask import Flask, request, jsonify
from earthquake_tsunami_predictor import TsunamiPredictor

# Create the Flask app
app = Flask(__name__)
app.config["DEBUG"] = True

# Create an instance of your predictor class
tsunami_predictor = TsunamiPredictor()

# Define the prediction endpoint
@app.route('/predict/', methods=['POST'])
def predict():
    """
    Accepts a JSON input with earthquake/tsunami features
    and returns a risk prediction as JSON.
    """
    prediction_input = request.get_json()
    return tsunami_predictor.predict_single_record(prediction_input)

# Optional health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "OK"}), 200

# Run the app
if __name__ == '__main__':
    app.run(port=int(os.getenv("PORT", 5000)), host='0.0.0.0', debug=True)
