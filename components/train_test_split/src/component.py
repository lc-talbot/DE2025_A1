import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(feature_path, train_path, test_path):
    """
    Split the dataset into training and testing sets (80/20 split).
    
    Args:
        feature_path: Path to input dataset CSV
        train_path: Path to save training data CSV
        test_path: Path to save testing data CSV
    """
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    # Read the input dataset
    logging.info(f'Reading data from {feature_path}')
    df = pd.read_csv(feature_path)
    logging.info(f'Dataset shape: {df.shape}')
    
    # Split the data (80% train, 20% test)
    logging.info('Splitting data with test_size=0.2, random_state=42')
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2,  # Hardcoded
        random_state=42,  # Hardcoded
        shuffle=True
    )
    
    logging.info(f'Training set shape: {train_df.shape}')
    logging.info(f'Test set shape: {test_df.shape}')
    
    # Create directories if they don't exist
    Path(train_path).parent.mkdir(parents=True, exist_ok=True)
    Path(test_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save the splits
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logging.info(f'Training data saved to {train_path}')
    logging.info(f'Test data saved to {test_path}')
    logging.info('Data split completed successfully!')


def parse_command_line_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_path', type=str, required=True,
                        help="Path to the input dataset CSV file")
    parser.add_argument('--train_path', type=str, required=True,
                        help="Path to save the training data CSV")
    parser.add_argument('--test_path', type=str, required=True,
                        help="Path to save the test data CSV")
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    split_data(**parse_command_line_arguments())