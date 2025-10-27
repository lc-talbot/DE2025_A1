#!/usr/bin/env python3
"""
Deployment Model Evaluation Component
Compares new model against currently deployed model.
Returns decision: DEPLOY_NEW or KEEP_OLD
"""

import argparse
import logging
import sys
import os
from pathlib import Path

import pandas as pd
import joblib
from google.cloud import storage
from sklearn.metrics import accuracy_score


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def download_deployed_model(project_id: str, bucket_name: str, model_file: str) -> str:
    """
    Download the currently deployed model from GCS.
    
    Returns:
        Local path to downloaded model, or None if doesn't exist
    """
    try:
        logging.info(f"Attempting to download deployed model: gs://{bucket_name}/{model_file}")
        
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(model_file)
        
        if not blob.exists():
            logging.info("No deployed model exists yet. This is the first deployment.")
            return None
        
        local_path = '/tmp/deployed_model.pkl'
        blob.download_to_filename(local_path)
        
        logging.info(f"Downloaded deployed model to {local_path}")
        return local_path
        
    except Exception as e:
        logging.error(f"Error downloading deployed model: {e}")
        return None


def evaluate_model(model_path: str, test_data_path: str) -> float:
    """
    Evaluate a model on test data.
    
    Returns:
        Accuracy score
    """
    logging.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    logging.info(f"Loading test data from {test_data_path}")
    test_data = pd.read_csv(test_data_path)
    
    X_test = test_data.drop('tsunami', axis=1)
    y_test = test_data['tsunami']
    
    logging.info("Making predictions...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Model accuracy: {accuracy:.4f}")
    
    return accuracy


def compare_models(project_id: str, bucket: str, deployed_model_file: str,
                  new_model_path: str, test_path: str, decision_path: str):
    """
    Compare new model against deployed model and make deployment decision.
    
    Writes decision to decision_path as a simple string:
    - "DEPLOY_NEW" if new model is better
    - "KEEP_OLD" if deployed model is better
    """
    
    logging.info("=" * 60)
    logging.info("DEPLOYMENT MODEL EVALUATION")
    logging.info("=" * 60)
    
    # Evaluate new model
    logging.info("Evaluating NEW model...")
    new_accuracy = evaluate_model(new_model_path, test_path)
    logging.info(f"New model accuracy: {new_accuracy:.4f}")
    
    # Try to download and evaluate deployed model
    deployed_model_path = download_deployed_model(project_id, bucket, deployed_model_file)
    
    decision = "DEPLOY_NEW"  # Default decision
    
    if deployed_model_path is None:
        # No deployed model exists - deploy the new one
        logging.info("=" * 60)
        logging.info("DECISION: DEPLOY_NEW")
        logging.info("REASON: No deployed model exists (first deployment)")
        logging.info("=" * 60)
        decision = "DEPLOY_NEW"
    else:
        # Evaluate deployed model
        logging.info("Evaluating DEPLOYED model...")
        deployed_accuracy = evaluate_model(deployed_model_path, test_path)
        logging.info(f"Deployed model accuracy: {deployed_accuracy:.4f}")
        
        # Compare
        improvement = new_accuracy - deployed_accuracy
        logging.info(f"Improvement: {improvement:+.4f}")
        
        if new_accuracy > deployed_accuracy:
            logging.info("=" * 60)
            logging.info("DECISION: DEPLOY_NEW")
            logging.info(f"REASON: New model is better ({new_accuracy:.4f} > {deployed_accuracy:.4f})")
            logging.info("=" * 60)
            decision = "DEPLOY_NEW"
        else:
            logging.info("=" * 60)
            logging.info("DECISION: KEEP_OLD")
            logging.info(f"REASON: Deployed model is better ({deployed_accuracy:.4f} >= {new_accuracy:.4f})")
            logging.info("=" * 60)
            decision = "KEEP_OLD"
    
    # Write decision to output file
    # CRITICAL: Write ONLY the decision string, nothing else!
    logging.info(f"Writing decision to {decision_path}")
    Path(decision_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(decision_path, 'w') as f:
        f.write(decision)  # Write ONLY "DEPLOY_NEW" or "KEEP_OLD"
    
    logging.info(f"Decision written: {decision}")
    logging.info("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', type=str, required=True)
    parser.add_argument('--bucket', type=str, required=True)
    parser.add_argument('--deployed_model_file', type=str, required=True)
    parser.add_argument('--new_model_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--decision_path', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    compare_models(
        project_id=args.project_id,
        bucket=args.bucket,
        deployed_model_file=args.deployed_model_file,
        new_model_path=args.new_model_path,
        test_path=args.test_path,
        decision_path=args.decision_path
    )