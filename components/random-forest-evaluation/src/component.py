import argparse
import json
import logging
import sys
import re
from pathlib import Path

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


def find_target_column(df, pattern='tsunami'):
    """
    Find the target column using regex pattern matching.
    
    Args:
        df: DataFrame to search
        pattern: Regex pattern to match (case-insensitive)
    
    Returns:
        Column name that matches the pattern
    
    Raises:
        ValueError: If no matching column found
    """
    regex = re.compile(pattern, re.IGNORECASE)
    matching_cols = [col for col in df.columns if regex.search(col)]
    
    if not matching_cols:
        raise ValueError(
            f"No column matching pattern '{pattern}' found. "
            f"Available columns: {df.columns.tolist()}"
        )
    
    if len(matching_cols) > 1:
        logging.warning(
            f"Multiple columns match pattern '{pattern}': {matching_cols}. "
            f"Using first match: '{matching_cols[0]}'"
        )
    
    return matching_cols[0]


def read_data_with_delimiter_detection(filepath):
    """
    Read CSV with automatic delimiter detection.
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        pandas DataFrame
    """
    # Try common delimiters
    for delimiter in [',', ';', '\t', '|']:
        try:
            df = pd.read_csv(filepath, delimiter=delimiter, nrows=5)
            if len(df.columns) > 1:
                logging.info(f"Detected delimiter: '{delimiter}'")
                return pd.read_csv(filepath, delimiter=delimiter)
        except Exception:
            continue
    
    # If nothing worked, default to comma
    logging.warning("Could not detect delimiter, defaulting to comma")
    return pd.read_csv(filepath)


def evaluate_random_forest(model_path, test_path, metrics_path):
    """
    Evaluate a Random Forest classifier on test data.
    
    Args:
        model_path: Path to the trained model (.pkl)
        test_path: Path to test data CSV
        metrics_path: Path to save metrics as JSON
    """
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    # Load the trained model
    logging.info(f'Loading Random Forest model from {model_path}')
    rf_model = joblib.load(model_path)
    logging.info('Model loaded successfully!')
    
    # Read test data with delimiter detection
    logging.info(f'Reading test data from {test_path}')
    test_df = read_data_with_delimiter_detection(test_path)
    logging.info(f'Test data shape: {test_df.shape}')
    logging.info(f'Column names: {test_df.columns.tolist()}')
    logging.info(f'Column dtypes:\n{test_df.dtypes}')
    
    # Find target column using regex
    target_col = find_target_column(test_df, pattern='tsunami')
    logging.info(f'Found target column: "{target_col}"')
    
    # Separate features and target
    X_test = test_df.drop(target_col, axis=1)
    y_test = test_df[target_col]
    
    logging.info(f'Features shape: {X_test.shape}')
    logging.info(f'Target shape: {y_test.shape}')
    
    # Make predictions
    logging.info('Making predictions on test data...')
    y_pred = rf_model.predict(X_test)
    
    # Calculate metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average='binary', zero_division=0))
    recall = float(recall_score(y_test, y_pred, average='binary', zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average='binary', zero_division=0))
    
    logging.info(f'Random Forest Accuracy: {accuracy:.4f}')
    logging.info(f'Random Forest Precision: {precision:.4f}')
    logging.info(f'Random Forest Recall: {recall:.4f}')
    logging.info(f'Random Forest F1 Score: {f1:.4f}')
    
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
    logging.info('Random Forest evaluation component finished successfully!')


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
    evaluate_random_forest(**parse_command_line_arguments())