import argparse
import logging
import sys
from pathlib import Path
import shutil

from google.cloud import storage
import requests


def upload_and_trigger(project_id, temp_bucket, model_bucket, model_file_name, 
                       new_model_path, cicd_webhook_url):
    """
    Upload approved model to temporary storage and trigger CI/CD pipeline.
    
    Args:
        project_id: GCP project ID
        temp_bucket: Temporary GCS bucket for model candidates
        model_bucket: Production GCS bucket for deployed models
        model_file_name: Name to save the model as in buckets
        new_model_path: Path to the approved new model
        cicd_webhook_url: Webhook URL to trigger CI/CD pipeline
    """
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    client = storage.Client(project=project_id)
    
    # Upload to temporary bucket (for CI/CD to test)
    logging.info(f'Uploading model to temporary bucket: gs://{temp_bucket}/{model_file_name}')
    temp_bucket_obj = client.get_bucket(temp_bucket)
    temp_blob = temp_bucket_obj.blob(model_file_name)
    temp_blob.upload_from_filename(new_model_path)
    logging.info('✅ Model uploaded to temporary storage successfully!')
    
    # Also upload to model bucket (backup/versioning)
    logging.info(f'Uploading model to model bucket: gs://{model_bucket}/{model_file_name}')
    model_bucket_obj = client.get_bucket(model_bucket)
    model_blob = model_bucket_obj.blob(model_file_name)
    model_blob.upload_from_filename(new_model_path)
    logging.info('✅ Model uploaded to model storage successfully!')
    
    # Trigger CI/CD pipeline
    if cicd_webhook_url and cicd_webhook_url != 'none':
        logging.info(f'\nTriggering CI/CD pipeline at: {cicd_webhook_url}')
        try:
            response = requests.post(
                cicd_webhook_url,
                json={'model_path': f'gs://{temp_bucket}/{model_file_name}'},
                timeout=10
            )
            if response.status_code in [200, 201, 204]:
                logging.info(f'✅ CI/CD pipeline triggered successfully! Status: {response.status_code}')
            else:
                logging.warning(f'⚠️ CI/CD trigger returned status: {response.status_code}')
                logging.warning(f'Response: {response.text}')
        except Exception as e:
            logging.error(f'❌ Failed to trigger CI/CD pipeline: {e}')
            logging.warning('Model was uploaded successfully, but CI/CD trigger failed.')
            logging.warning('You may need to trigger deployment manually.')
    else:
        logging.info('No CI/CD webhook URL provided - skipping trigger.')
        logging.info('Model uploaded successfully. Manual deployment may be required.')
    
    logging.info('\nModel upload and CI/CD trigger component finished successfully!')


def parse_command_line_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', type=str, required=True,
                        help="GCP project ID")
    parser.add_argument('--temp_bucket', type=str, required=True,
                        help="Temporary GCS bucket for model candidates")
    parser.add_argument('--model_bucket', type=str, required=True,
                        help="Production GCS bucket for deployed models")
    parser.add_argument('--model_file_name', type=str, default='model_candidate.pkl',
                        help="Name to save the model as in buckets")
    parser.add_argument('--new_model_path', type=str, required=True,
                        help="Path to the approved new model")
    parser.add_argument('--cicd_webhook_url', type=str, default='none',
                        help="Webhook URL to trigger CI/CD pipeline")
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    upload_and_trigger(**parse_command_line_arguments())