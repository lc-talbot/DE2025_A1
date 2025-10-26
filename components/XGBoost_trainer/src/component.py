import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import joblib
from xgboost import XGBClassifier


def train_xgboost(train_path, model_path):
    """
    Train an XGBoost classifier for tsunami prediction.
    
    Args:
        train_path: Path to training data CSV
        model_path: Path to save the trained model
    """
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    # Read training data
    logging.info(f'Reading training data from {train_path}')
    train_df = pd.read_csv(train_path)
    logging.info(f'Training data shape: {train_df.shape}')
    
    # Separate features and target
    X_train = train_df.drop('tsunami', axis=1)
    y_train = train_df['tsunami']
    
    logging.info(f'Features: {list(X_train.columns)}')
    logging.info(f'Target distribution:\n{y_train.value_counts()}')
    
    # Train XGBoost model
    logging.info('Training XGBoost classifier...')
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        n_jobs=-1  # Use all available cores
    )
    
    xgb_model.fit(X_train, y_train)
    logging.info('XGBoost training completed!')
    
    # Create directory if it doesn't exist
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save the model
    joblib.dump(xgb_model, model_path)
    logging.info(f'Model saved to {model_path}')
    logging.info('XGBoost trainer component finished successfully!')


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
    train_xgboost(**parse_command_line_arguments())