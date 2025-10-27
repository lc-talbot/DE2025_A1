#!/usr/bin/env python3
"""
Model Compare Component
Compares Random Forest and XGBoost models and selects the best one.
"""

import argparse
import json
import logging
import sys
import shutil
from pathlib import Path

import joblib


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def load_metrics(metrics_path: str) -> dict:
    """Load metrics from JSON file."""
    logging.info(f"Loading metrics from {metrics_path}")
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    logging.info(f"Loaded metrics: {metrics}")
    return metrics


def compare_and_select(rf_metrics_path: str, xgb_metrics_path: str,
                      rf_model_path: str, xgb_model_path: str,
                      best_model_path: str, best_model_name_path: str):
    """
    Compare models and select the best one based on accuracy.
    """
    
    logging.info("=" * 60)
    logging.info("MODEL COMPARISON")
    logging.info("=" * 60)
    
    # Load metrics
    rf_metrics = load_metrics(rf_metrics_path)
    xgb_metrics = load_metrics(xgb_metrics_path)
    
    rf_accuracy = rf_metrics['accuracy']
    xgb_accuracy = xgb_metrics['accuracy']
    
    logging.info(f"Random Forest accuracy: {rf_accuracy:.4f}")
    logging.info(f"XGBoost accuracy: {xgb_accuracy:.4f}")
    
    # Compare and select best
    if rf_accuracy >= xgb_accuracy:
        best_model = 'RandomForest'
        best_accuracy = rf_accuracy
        source_model_path = rf_model_path
        logging.info("=" * 60)
        logging.info("WINNER: Random Forest")
        logging.info(f"Accuracy: {best_accuracy:.4f}")
        logging.info("=" * 60)
    else:
        best_model = 'XGBoost'
        best_accuracy = xgb_accuracy
        source_model_path = xgb_model_path
        logging.info("=" * 60)
        logging.info("WINNER: XGBoost")
        logging.info(f"Accuracy: {best_accuracy:.4f}")
        logging.info("=" * 60)
    
    # Ensure output directories exist
    logging.info(f"Creating parent directory for: {best_model_path}")
    Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Creating parent directory for: {best_model_name_path}")
    Path(best_model_name_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Copy best model
    logging.info(f"Copying model from {source_model_path} to {best_model_path}")
    shutil.copy2(source_model_path, best_model_path)
    logging.info("Model copied successfully")
    
    # Write best model name
    logging.info(f"Writing best model name to {best_model_name_path}")
    with open(best_model_name_path, 'w') as f:
        f.write(best_model)
    logging.info(f"Best model name written: {best_model}")
    
    logging.info("=" * 60)
    logging.info("MODEL COMPARISON COMPLETE")
    logging.info("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rf_metrics_path', type=str, required=True)
    parser.add_argument('--xgb_metrics_path', type=str, required=True)
    parser.add_argument('--rf_model_path', type=str, required=True)
    parser.add_argument('--xgb_model_path', type=str, required=True)
    parser.add_argument('--best_model_path', type=str, required=True)
    parser.add_argument('--best_model_name_path', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    compare_and_select(
        rf_metrics_path=args.rf_metrics_path,
        xgb_metrics_path=args.xgb_metrics_path,
        rf_model_path=args.rf_model_path,
        xgb_model_path=args.xgb_model_path,
        best_model_path=args.best_model_path,
        best_model_name_path=args.best_model_name_path
    )