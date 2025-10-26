import argparse
import logging
import sys
from pathlib import Path
import os

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from google.cloud import storage


def evaluate_deployment_model(project_id, bucket, deployed_model_file, 
                               new_model_path, test_path, decision_path):
    """
    Compare the new best model against the currently deployed model.
    
    Args:
        project_id: GCP project ID
        bucket: GCS bucket name where deployed model is stored
        deployed_model_file: File name of currently deployed model in bucket
        new_model_path: Path to the new best model
        test_path: Path to test data CSV
        decision_path: Path to save deployment decision
    """
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    # Download currently deployed model from GCS
    logging.info(f'Downloading deployed model from gs://{bucket}/{deployed_model_file}')
    client = storage.Client(project=project_id)
    bucket_obj = client.get_bucket(bucket)
    blob = bucket_obj.blob(deployed_model_file)
    
    deployed_model_local_path = '/tmp/deployed_model.pkl'
    
    try:
        blob.download_to_filename(deployed_model_local_path)
        logging.info('✅ Deployed model downloaded successfully!')
        deployed_model_exists = True
    except Exception as e:
        logging.warning(f'⚠️ Could not download deployed model: {e}')
        logging.info('This might be the first deployment - no existing model to compare.')
        deployed_model_exists = False
    
    # Load new model
    logging.info(f'Loading new model from {new_model_path}')
    new_model = joblib.load(new_model_path)
    logging.info('New model loaded successfully!')
    
    # Read test data
    logging.info(f'Reading test data from {test_path}')
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop('tsunami', axis=1)
    y_test = test_df['tsunami']
    logging.info(f'Test data shape: {test_df.shape}')
    
    # Evaluate new model
    logging.info('Evaluating new model on test data...')
    y_pred_new = new_model.predict(X_test)
    new_accuracy = accuracy_score(y_test, y_pred_new)
    logging.info(f'New Model Accuracy: {new_accuracy:.4f}')
    
    # Decision logic
    if not deployed_model_exists:
        # First deployment - accept new model
        decision = 'DEPLOY_NEW'
        logging.info('\n=== DECISION ===')
        logging.info('✅ No deployed model exists. Deploying new model.')
        logging.info(f'New Model Accuracy: {new_accuracy:.4f}')
    else:
        # Load and evaluate deployed model
        logging.info(f'Loading deployed model from {deployed_model_local_path}')
        deployed_model = joblib.load(deployed_model_local_path)
        
        logging.info('Evaluating deployed model on test data...')
        y_pred_deployed = deployed_model.predict(X_test)
        deployed_accuracy = accuracy_score(y_test, y_pred_deployed)
        logging.info(f'Deployed Model Accuracy: {deployed_accuracy:.4f}')
        
        # Compare accuracies
        logging.info('\n=== MODEL COMPARISON ===')
        logging.info(f'Deployed Model Accuracy: {deployed_accuracy:.4f}')
        logging.info(f'New Model Accuracy:      {new_accuracy:.4f}')
        logging.info(f'Improvement:             {(new_accuracy - deployed_accuracy):.4f}')
        
        if new_accuracy > deployed_accuracy:
            decision = 'DEPLOY_NEW'
            logging.info('\n✅ DECISION: DEPLOY NEW MODEL')
            logging.info(f'New model is better by {(new_accuracy - deployed_accuracy):.4f}')
        else:
            decision = 'KEEP_OLD'
            logging.info('\n❌ DECISION: KEEP OLD MODEL')
            logging.info(f'Deployed model is better or equal')
    
    # Create directory if it doesn't exist
    Path(decision_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save decision
    with open(decision_path, 'w') as f:
        f.write(decision)
    
    logging.info(f'\nDecision saved to {decision_path}: {decision}')
    logging.info('Deployment model evaluation component finished successfully!')


def parse_command_line_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', type=str, required=True,
                        help="GCP project ID")
    parser.add_argument('--bucket', type=str, required=True,
                        help="GCS bucket name for deployed model")
    parser.add_argument('--deployed_model_file', type=str, required=True,
                        help="File name of deployed model in bucket")
    parser.add_argument('--new_model_path', type=str, required=True,
                        help="Path to the new best model")
    parser.add_argument('--test_path', type=str, required=True,
                        help="Path to test data CSV")
    parser.add_argument('--decision_path', type=str, required=True,
                        help="Path to save deployment decision")
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    evaluate_deployment_model(**parse_command_line_arguments())