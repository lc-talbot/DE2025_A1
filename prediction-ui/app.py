import os
import json
import requests
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)
app.config["DEBUG"] = os.environ.get('DEBUG', 'False') == 'True'

# Use your Cloud Run URL by default; can still override via env
PREDICTOR_API_URL = os.environ.get(
    "PREDICTOR_API_URL",
    "https://tsunami-prediction-api-861179434993.us-central1.run.app"
).rstrip("/")

# The exact feature names your model expects
FEATURES = ["magnitude", "cdi", "depth", "latitude", "longitude"]

@app.route('/', methods=["GET", "POST"])
def assess_risk():
    if request.method == "GET":
        return render_template("input_form_page.html")

    # POST -> collect inputs and call predictor
    try:
        form_data = {}
        for name in FEATURES:
            raw = request.form.get(name, "")
            # try numeric conversion
            try:
                form_data[name] = float(raw)
            except (TypeError, ValueError):
                # keep raw (shouldn't happen for numeric fields, but safe)
                form_data[name] = raw

        # Build payload as a list of records
        payload = [form_data]

        res = requests.post(
            f"{PREDICTOR_API_URL}/predict/",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        if res.status_code != 200:
            # try to surface API error
            err_json = {}
            try:
                if "application/json" in res.headers.get("content-type", ""):
                    err_json = res.json()
            except Exception:
                pass
            err_msg = err_json.get("error", f"API returned status {res.status_code}")
            return render_template(
                "response_page.html",
                error=True,
                error_message=err_msg,
                details=err_json
            )

        api_response = res.json()
        prediction_value = api_response.get("prediction")
        tsunami_risk = api_response.get("tsunami_risk", "Unknown")
        probability = api_response.get("probability", 0.0)
        message = api_response.get("message", "")

        probability_percent = round(probability * 100, 1) if probability else 0

        return render_template(
            "response_page.html",
            error=False,
            prediction=prediction_value,
            risk_level=tsunami_risk,
            probability=probability,
            probability_percent=probability_percent,
            message=message,
            input_data=form_data
        )

    except requests.exceptions.ConnectionError:
        return render_template(
            "response_page.html",
            error=True,
            error_message="Cannot connect to prediction API",
            details={"api_url": PREDICTOR_API_URL}
        )
    except requests.exceptions.Timeout:
        return render_template(
            "response_page.html",
            error=True,
            error_message="Prediction API timeout",
            details={"message": "The request took too long"}
        )
    except Exception as e:
        return render_template(
            "response_page.html",
            error=True,
            error_message=f"An error occurred: {str(e)}",
            details={}
        )

@app.route('/health', methods=['GET'])
def health():
    # Check predictor health at /healthz
    api_healthy = False
    try:
        r = requests.get(f"{PREDICTOR_API_URL}/healthz", timeout=5)
        api_healthy = r.status_code == 200
    except Exception:
        pass

    return jsonify({
        "status": "healthy",
        "ui": "OK",
        "api_url": PREDICTOR_API_URL,
        "api_reachable": api_healthy
    }), 200

@app.route('/api-info', methods=['GET'])
def api_info():
    try:
        r = requests.get(f"{PREDICTOR_API_URL}/healthz", timeout=10)
        if r.status_code == 200:
            return jsonify(r.json()), 200
        return jsonify({"error": "API not available"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 503

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    print(f"Starting UI server on port {port}")
    print(f"Prediction API URL: {PREDICTOR_API_URL}")
    app.run(port=port, host='0.0.0.0', debug=app.config["DEBUG"])