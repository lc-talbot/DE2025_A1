import json
import os
import pandas as pd
import joblib
from flask import jsonify
from io import StringIO, BytesIO
from google.cloud import storage

class TsunamiPredictor:
    def __init__(self):
        """
        Initialize the predictor and load the model from Cloud Storage
        """
        self.model = None
        self.feature_names = None  # Store expected feature names from training
        self.model_bucket = os.environ.get('MODEL_BUCKET', 'models_tsunami_2023019')
        self.model_file = os.environ.get('MODEL_FILE', 'deployed_model.pkl')
        self.project_id = os.environ.get('PROJECT_ID', 'tsunami-prediction-mlops')
        
        # Load the model at startup
        self._load_model()

    def _load_model(self):
        """
        Load the trained model from Google Cloud Storage
        """
        try:
            print(f"Loading model from gs://{self.model_bucket}/{self.model_file}")
            
            # Initialize Cloud Storage client
            client = storage.Client(project=self.project_id)
            bucket = client.bucket(self.model_bucket)
            blob = bucket.blob(self.model_file)
            
            # Download model as bytes and load with joblib
            model_bytes = blob.download_as_bytes()
            self.model = joblib.load(BytesIO(model_bytes))
            
            # Extract feature names if available (sklearn models store this)
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = list(self.model.feature_names_in_)
                print(f"✓ Model expects features: {self.feature_names}")
            else:
                print("⚠ Model does not have feature_names_in_ attribute")
                print("  Make sure input JSON has features in the correct order")
            
            print(f"✓ Model loaded successfully: {type(self.model).__name__}")
            
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")
            print("API will return mock predictions until model is available")
            self.model = None
            self.feature_names = None

    def predict_single_record(self, prediction_input):
        """
        Handles a single JSON prediction request.
        
        Input example:
        [
          {
            "feature1": value1,
            "feature2": value2,
            ...
            (Features must match training data columns, excluding 'tsunami')
          }
        ]
        
        Output example:
        {
          "prediction": 1,
          "tsunami_risk": "High",
          "probability": 0.87,
          "model_type": "RandomForestClassifier"
        }
        """
        
        print("Received prediction request:", prediction_input)

        # Convert JSON → DataFrame
        df = pd.read_json(StringIO(json.dumps(prediction_input)), orient='records')
        
        # If model not loaded, return mock prediction
        if self.model is None:
            return jsonify({
                'error': 'Model not available',
                'prediction': None,
                'tsunami_risk': 'Unknown',
                'message': 'Model is not loaded. Please check server logs.'
            }), 503
        
        try:
            # Validate and reorder features if model has feature names
            if self.feature_names is not None:
                # Check if all required features are present
                missing_features = set(self.feature_names) - set(df.columns)
                if missing_features:
                    return jsonify({
                        'error': 'Missing features',
                        'missing': list(missing_features),
                        'expected_features': self.feature_names,
                        'received_features': list(df.columns)
                    }), 400
                
                # Reorder DataFrame columns to match training order
                df = df[self.feature_names]
                print(f"Features reordered to match training: {list(df.columns)}")
            else:
                print(f"Using features as provided: {list(df.columns)}")
            
            # Make prediction with the loaded model
            y_pred = self.model.predict(df)[0]
            
            # Get probability if available
            if hasattr(self.model, 'predict_proba'):
                y_proba = self.model.predict_proba(df)[0]
                tsunami_probability = float(y_proba[1])  # Probability of tsunami (class 1)
            else:
                tsunami_probability = float(y_pred)
            
            # Map prediction to risk level
            if y_pred == 1:
                risk_level = "High" if tsunami_probability > 0.7 else "Medium"
            else:
                risk_level = "Low" if tsunami_probability < 0.3 else "Medium"
            
            # Return structured response
            response = {
                'prediction': int(y_pred),
                'tsunami_risk': risk_level,
                'probability': round(tsunami_probability, 4),
                'model_type': type(self.model).__name__,
                'message': 'Tsunami likely' if y_pred == 1 else 'No tsunami expected',
                'features_used': list(df.columns)
            }
            
            print(f"Prediction result: {response}")
            return jsonify(response), 200
            
        except Exception as e:
            error_message = f"Error during prediction: {str(e)}"
            print(error_message)
            import traceback
            traceback.print_exc()
            return jsonify({
                'error': error_message,
                'prediction': None,
                'expected_features': self.feature_names if self.feature_names else 'unknown'
            }), 500

    def reload_model(self):
        """
        Reload the model from Cloud Storage.
        Useful for updating the model without restarting the service.
        """
        print("Reloading model...")
        self._load_model()
        return self.model is not None