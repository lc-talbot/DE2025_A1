import argparse
import logging
import sys
import re
from pathlib import Path

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier


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


def train_random_forest(train_path, model_path):
    """
    Train a Random Forest classifier for tsunami prediction.
    
    Args:
        train_path: Path to training data CSV
        model_path: Path to save the trained model
    """
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    # Read training data
    logging.info(f'Reading training data from {train_path}')
    train_df = pd.read_csv(train_path, sep=';')
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
    
    # Train Random Forest model
    logging.info('Training Random Forest classifier...')
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    rf_model.fit(X_train, y_train)
    logging.info('Random Forest training completed!')
    
    # Create directory if it doesn't exist
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save the model
    joblib.dump(rf_model, model_path)
    logging.info(f'Model saved to {model_path}')
    logging.info('Random Forest trainer component finished successfully!')


def parse_command_line_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True,
                        help="Path to the training data CSV file")
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to save the trained model (.pkl)")
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    train_random_forest(**parse_command_line_arguments())