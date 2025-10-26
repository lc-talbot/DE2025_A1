import argparse
import logging
import sys
from pathlib import Path
import shutil


def compare_models(rf_metrics_path, xgb_metrics_path, rf_model_path, xgb_model_path, 
                   best_model_path, best_model_name_path):
    """
    Compare Random Forest and XGBoost models and select the best one.
    
    Args:
        rf_metrics_path: Path to RF accuracy file
        xgb_metrics_path: Path to XGBoost accuracy file
        rf_model_path: Path to RF model file
        xgb_model_path: Path to XGBoost model file
        best_model_path: Path to save the best model
        best_model_name_path: Path to save the best model name
    """
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    # Read RF accuracy
    logging.info(f'Reading Random Forest accuracy from {rf_metrics_path}')
    with open(rf_metrics_path, 'r') as f:
        rf_accuracy = float(f.read().strip())
    logging.info(f'Random Forest Accuracy: {rf_accuracy:.4f}')
    
    # Read XGBoost accuracy
    logging.info(f'Reading XGBoost accuracy from {xgb_metrics_path}')
    with open(xgb_metrics_path, 'r') as f:
        xgb_accuracy = float(f.read().strip())
    logging.info(f'XGBoost Accuracy: {xgb_accuracy:.4f}')
    
    # Compare and select best model
    logging.info('\n=== Model Comparison ===')
    if rf_accuracy > xgb_accuracy:
        best_model = 'RandomForest'
        best_accuracy = rf_accuracy
        source_model_path = rf_model_path
        logging.info(f'✅ Random Forest wins with accuracy: {rf_accuracy:.4f}')