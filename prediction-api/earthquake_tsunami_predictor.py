import json
import os
import pandas as pd
from flask import jsonify
from io import StringIO

class TsunamiPredictor:
    def __init__(self):
        # Later load trained model here
        self.model = None

    def predict_single_record(self, prediction_input):
        """
        Handles a single JSON prediction request.
        Input example:
        [
          {
            "Magnitude": 7.8,
            "Depth": 25.0,
            "DistanceToCoast": 120.5
          }
        ]
        """

        print("Received input:", prediction_input)

        # Convert JSON → DataFrame
        df = pd.read_json(StringIO(json.dumps(prediction_input)), orient='records')

        # If model not loaded yet, return mock prediction
        if self.model is None:
            result = "Medium Risk (Mock Prediction)"
            return jsonify({'prediction': result}), 200

        # Example for later (when you have a real model):
        # y_pred = self.model.predict(df)[0]
        # result = str(y_pred)
        # return jsonify({'prediction': result}), 200
