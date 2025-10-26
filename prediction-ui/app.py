import os
import json
import requests
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def assess_risk():
    # If user just visits the page, show the input form
    if request.method == "GET":
        return render_template("input_form_page.html")

    # If the user submits the form, do a prediction
    elif request.method == "POST":
        # Collect form inputs from the HTML form
        # (These names must match the <input name="..."> fields in your HTML form)
        prediction_input = [
            {
                "Magnitude": float(request.form.get("Magnitude")),
                "Depth": float(request.form.get("Depth")),
                "DistanceToCoast": float(request.form.get("DistanceToCoast"))
            }
        ]

        # Call the prediction API (your container from prediction-api)
        predictor_api_url = os.environ["PREDICTOR_API"]  # e.g. http://<vm-ip>:5000/predict/
        res = requests.post(predictor_api_url, json=json.loads(json.dumps(prediction_input)))

        # Get result from API response
        prediction_value = res.json()["prediction"]

        # Render a response page for the user
        return render_template(
            "response_page.html",
            risk_level=prediction_value
        )

    # Block any other HTTP methods
    else:
        return jsonify(message="Method Not Allowed"), 405

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "UI OK"}), 200

if __name__ == '__main__':
    # UI can run on port 5001 so it doesn't collide with the API on 5000
    app.run(port=int(os.environ.get("PORT", 5001)), host='0.0.0.0', debug=True)
