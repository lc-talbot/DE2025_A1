import logging
import sys
from pathlib import Path
import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from google.cloud import storage
from kfp.v2.dsl import component, Input, Output, Dataset, Artifact, Model


@component(
    base_image="python:3.10",
    packages_to_install=[
        "pandas==1.5.3",
        "scikit-learn==1.2.2",
        "joblib==1.3.2",
        "google-cloud-storage==2.10.0",
    ],
)
def evaluate_deployment_model(
    project_id: str,
    bucket: str,
    deployed_model_file: str,
    new_model: Input[Model],
    test_data: Input[Dataset],
    decision: Output[Artifact],
):
    """
    Compare the new best model against the currently deployed model on GCS.
    Writes 'DEPLOY_NEW' or 'KEEP_OLD' to the decision artifact.
    """

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger("evaluate_deployment_model")

    deployed_model_local_path = "/tmp/deployed_model.pkl"

    # --- Download deployed model from GCS ---
    logger.info(f"Downloading deployed model from gs://{bucket}/{deployed_model_file}")
    client = storage.Client(project=project_id)
    bucket_obj = client.bucket(bucket)
    blob = bucket_obj.blob(deployed_model_file)

    try:
        blob.download_to_filename(deployed_model_local_path)
        logger.info("✅ Deployed model downloaded successfully!")
        deployed_model_exists = True
    except Exception as e:
        logger.warning(f"⚠️ Could not download deployed model: {e}")
        logger.info("This might be the first deployment - no existing model to compare.")
        deployed_model_exists = False

    # --- Load new model ---
    new_model_path = os.path.join(new_model.path, "model.joblib")
    logger.info(f"Loading new model from {new_model_path}")
    new_model_obj = joblib.load(new_model_path)
    logger.info("New model loaded successfully!")

    # --- Read test data ---
    logger.info(f"Reading test data from {test_data.path}")
    test_df = pd.read_csv(test_data.path)
    X_test = test_df.drop("tsunami", axis=1)
    y_test = test_df["tsunami"]
    logger.info(f"Test data shape: {test_df.shape}")

    # --- Evaluate new model ---
    logger.info("Evaluating new model on test data...")
    y_pred_new = new_model_obj.predict(X_test)
    new_accuracy = accuracy_score(y_test, y_pred_new)
    logger.info(f"New Model Accuracy: {new_accuracy:.4f}")

    # --- Decision logic ---
    if not deployed_model_exists:
        decision_value = "DEPLOY_NEW"
        logger.info("✅ No deployed model exists. Deploying new model.")
    else:
        logger.info(f"Loading deployed model from {deployed_model_local_path}")
        deployed_model = joblib.load(deployed_model_local_path)

        logger.info("Evaluating deployed model on test data...")
        y_pred_deployed = deployed_model.predict(X_test)
        deployed_accuracy = accuracy_score(y_test, y_pred_deployed)
        logger.info(f"Deployed Model Accuracy: {deployed_accuracy:.4f}")

        improvement = new_accuracy - deployed_accuracy
        logger.info(f"Accuracy improvement: {improvement:.4f}")

        if new_accuracy > deployed_accuracy:
            decision_value = "DEPLOY_NEW"
            logger.info("✅ DECISION: DEPLOY NEW MODEL")
        else:
            decision_value = "KEEP_OLD"
            logger.info("❌ DECISION: KEEP OLD MODEL")

    # --- Save decision artifact ---
    Path(decision.path).parent.mkdir(parents=True, exist_ok=True)
    with open(decision.path, "w") as f:
        f.write(decision_value)

    logger.info(f"Decision saved to {decision.path}: {decision_value}")
    logger.info("Deployment model evaluation completed successfully.")
