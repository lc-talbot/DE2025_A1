import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


def evaluate_xgboost(model_path, test_path, metrics_path):
    """
    Evaluate an XGBoost classifier on test data.
    
    Args:
        model_path: Path to the trained model (.pkl)
        test_path: Path to test data CSV
        metrics_path: Path to save metrics as JSON
    """
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    # Load the trained model
    logging.info(f'Loading XGBoost model from {model_path}')
    xgb_model = joblib.load(model_path)
    logging.info('Model loaded successfully!')
    
    # Read test data
    logging.info(f'Reading test data from {test_path}')
    test_df = pd.read_csv(test_path)
    logging.info(f'Test data shape: {test_df.shape}')
    
    # Separate features and target
    X_test = test_df.drop('tsunami', axis=1)
    y_test = test_df['tsunami']
    
    # Make predictions
    logging.info('Making predictions on test data...')
    y_pred = xgb_model.predict(X_test)
    
    # Calculate metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average='binary', zero_division=0))
    recall = float(recall_score(y_test, y_pred, average='binary', zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average='binary', zero_division=0))
    
    logging.info(f'XGBoost Accuracy: {accuracy:.4f}')
    logging.info(f'XGBoost Precision: {precision:.4f}')
    logging.info(f'XGBoost Recall: {recall:.4f}')
    logging.info(f'XGBoost F1 Score: {f1:.4f}')
    
    # Detailed metrics
    logging.info('\n=== Classification Report ===')
    logging.info('\n' + classification_report(y_test, y_pred, 
                                               target_names=['No Tsunami', 'Tsunami']))
    
    logging.info('\n=== Confusion Matrix ===')
    cm = confusion_matrix(y_test, y_pred)
    logging.info(f'\n{cm}')
    logging.info(f'True Negatives: {cm[0][0]}, False Positives: {cm[0][1]}')
    logging.info(f'False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}')
    
    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    # Create directory if it doesn't exist
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save metrics as JSON (for pipeline to use)
    logging.info(f'Saving metrics to {metrics_path}')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logging.info(f'Metrics saved: {metrics}')
    logging.info('XGBoost evaluation component finished successfully!')


def parse_command_line_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to the trained model (.pkl)")
    parser.add_argument('--test_path', type=str, required=True,
                        help="Path to the test data CSV file")
    parser.add_argument('--metrics_path', type=str, required=True,
                        help="Path to save the metrics JSON file")
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    evaluate_xgboost(**parse_command_line_arguments())