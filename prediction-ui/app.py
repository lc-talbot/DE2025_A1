import os
import json
import requests
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)
app.config["DEBUG"] = os.environ.get('DEBUG', 'False') == 'True'

# Get the prediction API URL from environment variable
PREDICTOR_API_URL = os.environ.get("PREDICTOR_API_URL", "http://localhost:5000")

@app.route('/', methods=["GET", "POST"])
def assess_risk():
    """
    Main route - shows form on GET, processes prediction on POST
    """
    # If user just visits the page, show the input form
    if request.method == "GET":
        return render_template("input_form_page.html")

    # If the user submits the form, do a prediction
    elif request.method == "POST":
        try:
            # Collect form inputs from the HTML form
            # Build the prediction input based on form data
            form_data = {}
            for key in request.form:
                try:
                    # Try to convert to float
                    form_data[key] = float(request.form.get(key))
                except (ValueError, TypeError):
                    # If not a number, keep as string
                    form_data[key] = request.form.get(key)
            
            prediction_input = [form_data]
            
            print(f"Sending to API: {prediction_input}")
            print(f"API URL: {PREDICTOR_API_URL}/predict/")
            
            # Call the prediction API
            res = requests.post(
                f"{PREDICTOR_API_URL}/predict/",
                json=prediction_input,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            # Check if request was successful
            if res.status_code != 200:
                error_data = res.json() if res.headers.get('content-type') == 'application/json' else {}
                error_msg = error_data.get('error', f'API returned status {res.status_code}')
                return render_template(
                    "response_page.html",
                    error=True,
                    error_message=error_msg,
                    details=error_data
                )
            
            # Get result from API response
            api_response = res.json()
            print(f"API Response: {api_response}")
            
            # Extract prediction details
            prediction_value = api_response.get("prediction")
            tsunami_risk = api_response.get("tsunami_risk", "Unknown")
            probability = api_response.get("probability", 0.0)
            message = api_response.get("message", "")
            
            # Convert probability to percentage for display
            probability_percent = round(probability * 100, 1) if probability else 0
            
            # Render a response page for the user
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
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return render_template(
                "response_page.html",
                error=True,
                error_message=f"An error occurred: {str(e)}",
                details={}
            )

    # Block any other HTTP methods
    else:
        return jsonify(message="Method Not Allowed"), 405

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint
    """
    # Try to check if API is reachable
    api_healthy = False
    try:
        api_response = requests.get(f"{PREDICTOR_API_URL}/health", timeout=5)
        api_healthy = api_response.status_code == 200
    except:
        pass
    
    return jsonify({
        "status": "healthy",
        "ui": "OK",
        "api_url": PREDICTOR_API_URL,
        "api_reachable": api_healthy
    }), 200

@app.route('/api-info', methods=['GET'])
def api_info():
    """
    Get information about the prediction API
    """
    try:
        health_response = requests.get(f"{PREDICTOR_API_URL}/health", timeout=10)
        if health_response.status_code == 200:
            return jsonify(health_response.json()), 200
        else:
            return jsonify({"error": "API not available"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 503

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    print(f"Starting UI server on port {port}")
    print(f"Prediction API URL: {PREDICTOR_API_URL}")
    app.run(port=port, host='0.0.0.0', debug=app.config["DEBUG"])