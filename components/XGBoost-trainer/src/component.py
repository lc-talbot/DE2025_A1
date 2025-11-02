import argparse
import logging
import sys
import re
from pathlib import Path

import pandas as pd
import joblib
from google.cloud import storage

import xgboost as xgb


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


def upload_model_to_gcs(local_path, bucket_name, blob_name):
    """
    Upload model to Google Cloud Storage.
    
    Args:
        local_path: Local path to model file
        bucket_name: GCS bucket name
        blob_name: Path within bucket (e.g., 'models/candidate_model_xgb.pkl')
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)
        logging.info(f"✓ Model uploaded to gs://{bucket_name}/{blob_name}")
        return True
    except Exception as e:
        logging.error(f"✗ Failed to upload model to GCS: {e}")
        return False


def train_xgboost(train_path, model_path, gcs_model_path=None):
    """
    Train an XGBoost classifier for tsunami prediction.
    
    Args:
        train_path: Path to training data CSV
        model_path: Local path to save the trained model
        gcs_model_path: Optional GCS path (gs://bucket/path) to upload model
    """
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    # Read training data with delimiter detection
    logging.info(f'Reading training data from {train_path}')
    train_df = read_data_with_delimiter_detection(train_path)
    logging.info(f'Training data shape: {train_df.shape}')
    logging.info(f'Column names: {train_df.columns.tolist()}')
    logging.info(f'Column dtypes:\n{train_df.dtypes}')
    logging.info(f'Null values:\n{train_df.isnull().sum()}')
    
    # Find target column using regex
    target_col = find_target_column(train_df, pattern='tsunami')
    logging.info(f'Found target column: "{target_col}"')
    
    # Separate features and target
    X_train = train_df.drop(target_col, axis=1)
    y_train = train_df[target_col]
    
    logging.info(f'Features: {list(X_train.columns)}')
    logging.info(f'Target distribution:\n{y_train.value_counts()}')
    
    # Train XGBoost model
    logging.info('Training XGBoost classifier...')
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,  # Use all available cores
        verbosity=1
    )
    
    xgb_model.fit(X_train, y_train)
    logging.info('XGBoost training completed!')
    
    # Create directory if it doesn't exist
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save the model locally
    joblib.dump(xgb_model, model_path)
    logging.info(f'Model saved to {model_path}')
    
    # Upload to GCS if path provided
    if gcs_model_path:
        gcs_parts = gcs_model_path.replace("gs://", "").split("/", 1)
        if len(gcs_parts) == 2:
            bucket_name, blob_name = gcs_parts
            upload_model_to_gcs(model_path, bucket_name, blob_name)
    
    logging.info('XGBoost trainer component finished successfully!')


def parse_command_line_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True,
                        help="Path to the training data CSV file")
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to save the trained model (.pkl)")
    parser.add_argument('--gcs_model_path', type=str, required=False,
                        help="Optional GCS path to upload model (gs://bucket/path)")
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    train_xgboost(**parse_command_line_arguments())