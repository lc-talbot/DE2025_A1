import google.cloud.aiplatform as aip
import kfp
from kfp import compiler
from kfp import dsl
from kfp.dsl import (Artifact,
                     Input,
                     Output,
                     Model,
                     Dataset,
                     Metrics,
                     OutputPath)
import os


# ==================== CONFIGURATION ====================
# These can be set via environment variables or defaults
PROJECT_ID = os.getenv('PROJECT_ID', 'your-project-id')
REGION = os.getenv('REGION', 'us-central1')
REPOSITORY = os.getenv('REPOSITORY', 'tsunami-repo')

# Buckets
DATA_BUCKET = os.getenv('DATA_BUCKET', 'data_tsunami_2023019')
MODEL_BUCKET = os.getenv('MODEL_BUCKET', 'models_tsunami_2023019')
TEMP_BUCKET = os.getenv('TEMP_BUCKET', 'temp_tsunami_2023019')


# ==================== COMPONENT DEFINITIONS ====================

@dsl.container_component
def data_ingestion(project: str, bucket: str, data_file_name: str, dataset: Output[Dataset]):
    return dsl.ContainerSpec(
        image=f'{REGION}-docker.pkg.dev/{project}/{REPOSITORY}/data-ingestion:0.0.1',
        command=['python3', '/pipelines/component/src/component.py'],
        args=['--project_id', project, '--bucket', bucket, '--file_name', data_file_name, '--feature_path',
              dataset.path])


@dsl.container_component
def train_test_split(project: str, dataset: Input[Dataset], train_data: Output[Dataset], test_data: Output[Dataset]):
    return dsl.ContainerSpec(
        image=f'{REGION}-docker.pkg.dev/{project}/{REPOSITORY}/train-test-split:0.0.1',
        command=['python3', '/pipelines/component/src/component.py'],
        args=['--feature_path', dataset.path, '--train_path', train_data.path, '--test_path', test_data.path])


@dsl.container_component
def random_forest_trainer(project: str, train_data: Input[Dataset], rf_model: Output[Model]):
    return dsl.ContainerSpec(
        image=f'{REGION}-docker.pkg.dev/{project}/{REPOSITORY}/random-forest-trainer:0.0.1',
        command=['python3', '/pipelines/component/src/component.py'],
        args=['--train_path', train_data.path, '--model_path', rf_model.path])


@dsl.container_component
def xgboost_trainer(project: str, train_data: Input[Dataset], xgb_model: Output[Model]):
    return dsl.ContainerSpec(
        image=f'{REGION}-docker.pkg.dev/{project}/{REPOSITORY}/xgboost-trainer:0.0.1',
        command=['python3', '/pipelines/component/src/component.py'],
        args=['--train_path', train_data.path, '--model_path', xgb_model.path])


@dsl.container_component
def random_forest_evaluation(project: str, rf_model: Input[Model], test_data: Input[Dataset], rf_metrics: Output[Metrics]):
    return dsl.ContainerSpec(
        image=f'{REGION}-docker.pkg.dev/{project}/{REPOSITORY}/random-forest-evaluation:0.0.1',
        command=['python3', '/pipelines/component/src/component.py'],
        args=['--model_path', rf_model.path, '--test_path', test_data.path, '--metrics_path', rf_metrics.path])


@dsl.container_component
def xgboost_evaluation(project: str, xgb_model: Input[Model], test_data: Input[Dataset], xgb_metrics: Output[Metrics]):
    return dsl.ContainerSpec(
        image=f'{REGION}-docker.pkg.dev/{project}/{REPOSITORY}/xgboost-evaluation:0.0.1',
        command=['python3', '/pipelines/component/src/component.py'],
        args=['--model_path', xgb_model.path, '--test_path', test_data.path, '--metrics_path', xgb_metrics.path])


@dsl.container_component
def model_compare(project: str, rf_metrics: Input[Metrics], xgb_metrics: Input[Metrics], 
                  rf_model: Input[Model], xgb_model: Input[Model],
                  best_model: Output[Model], best_model_name: Output[Artifact]):
    return dsl.ContainerSpec(
        image=f'{REGION}-docker.pkg.dev/{project}/{REPOSITORY}/model-compare:0.0.1',
        command=['python3', '/pipelines/component/src/component.py'],
        args=['--rf_metrics_path', rf_metrics.path, '--xgb_metrics_path', xgb_metrics.path,
              '--rf_model_path', rf_model.path, '--xgb_model_path', xgb_model.path,
              '--best_model_path', best_model.path, '--best_model_name_path', best_model_name.path])


@dsl.container_component
def deployment_model_evaluation(project: str, bucket: str, deployed_model_file: str,
                                best_model: Input[Model], test_data: Input[Dataset],
                                decision: OutputPath(str)):
    return dsl.ContainerSpec(
        image=f'{REGION}-docker.pkg.dev/{project}/{REPOSITORY}/deployment-model-evaluation:0.0.1',
        command=['python3', '/pipelines/component/src/component.py'],
        args=['--project_id', project, '--bucket', bucket, '--deployed_model_file', deployed_model_file,
              '--new_model_path', best_model.path, '--test_path', test_data.path, '--decision_path', decision])


@dsl.container_component
def model_upload_trigger_cicd(project: str, temp_bucket: str, model_bucket: str, 
                              model_file_name: str, best_model: Input[Model], cicd_webhook_url: str):
    return dsl.ContainerSpec(
        image=f'{REGION}-docker.pkg.dev/{project}/{REPOSITORY}/model-upload-trigger-cicd:0.0.1',
        command=['python3', '/pipelines/component/src/component.py'],
        args=['--project_id', project, '--temp_bucket', temp_bucket, '--model_bucket', model_bucket,
              '--model_file_name', model_file_name, '--new_model_path', best_model.path,
              '--cicd_webhook_url', cicd_webhook_url])


# ==================== PIPELINE DEFINITION ====================

@kfp.dsl.pipeline(
    name="tsunami-training-pipeline")
def tsunami_pipeline(project_id: str, data_bucket: str, data_filename: str,
                     model_bucket: str, temp_bucket: str, deployed_model_file: str,
                     model_candidate_file: str, cicd_webhook_url: str):
    """
    Tsunami Prediction Training Pipeline
    
    Trains Random Forest and XGBoost models to predict tsunami risk from earthquake data.
    Automatically selects and deploys the best performing model.
    """
    
    # Step 1: Data Ingestion
    di_op = data_ingestion(
        project=project_id,
        bucket=data_bucket,
        data_file_name=data_filename
    )

    # Step 2: Train-Test Split
    split_op = train_test_split(
        project=project_id,
        dataset=di_op.outputs['dataset']
    )
    
    # Step 3a: Train Random Forest
    rf_train_op = random_forest_trainer(
        project=project_id,
        train_data=split_op.outputs['train_data']
    )
    
    # Step 3b: Train XGBoost
    xgb_train_op = xgboost_trainer(
        project=project_id,
        train_data=split_op.outputs['train_data']
    )
    
    # Step 4a: Evaluate Random Forest
    rf_eval_op = random_forest_evaluation(
        project=project_id,
        rf_model=rf_train_op.outputs['rf_model'],
        test_data=split_op.outputs['test_data']
    )
    
    # Step 4b: Evaluate XGBoost
    xgb_eval_op = xgboost_evaluation(
        project=project_id,
        xgb_model=xgb_train_op.outputs['xgb_model'],
        test_data=split_op.outputs['test_data']
    )
    
    # Step 5: Compare Models
    compare_op = model_compare(
        project=project_id,
        rf_metrics=rf_eval_op.outputs['rf_metrics'],
        xgb_metrics=xgb_eval_op.outputs['xgb_metrics'],
        rf_model=rf_train_op.outputs['rf_model'],
        xgb_model=xgb_train_op.outputs['xgb_model']
    )
    
    # Step 6: Evaluate Against Deployed Model
    deployment_eval_op = deployment_model_evaluation(
        project=project_id,
        bucket=model_bucket,
        deployed_model_file=deployed_model_file,
        best_model=compare_op.outputs['best_model'],
        test_data=split_op.outputs['test_data']
    )
    
    # Step 7: Upload Model and Trigger CI/CD (conditional on approval)
    # Using dsl.Condition to check if decision is "DEPLOY_NEW"
    with dsl.Condition(
        deployment_eval_op.outputs['decision'] == "DEPLOY_NEW",
        name='check-deployment-approval'
    ):
        upload_op = model_upload_trigger_cicd(
            project=project_id,
            temp_bucket=temp_bucket,
            model_bucket=model_bucket,
            model_file_name=model_candidate_file,
            best_model=compare_op.outputs['best_model'],
            cicd_webhook_url=cicd_webhook_url
        )


# ==================== COMPILE FUNCTION ====================

def compile_pipeline():
    """Compile the pipeline to YAML format."""
    compiler.Compiler().compile(
        pipeline_func=tsunami_pipeline,
        package_path='tsunami_training_pipeline.yaml')
    print('✅ Pipeline compiled successfully to: tsunami_training_pipeline.yaml')


# ==================== RUN FUNCTION ====================

def run_pipeline():
    """Submit the pipeline to Vertex AI for execution."""
    
    # Get configuration from environment or use defaults
    project_id = os.getenv('PROJECT_ID', PROJECT_ID)
    region = os.getenv('REGION', REGION)
    data_bucket = os.getenv('DATA_BUCKET', DATA_BUCKET)
    model_bucket = os.getenv('MODEL_BUCKET', MODEL_BUCKET)
    temp_bucket = os.getenv('TEMP_BUCKET', TEMP_BUCKET)
    
    # Pipeline root - where pipeline artifacts are stored
    pipeline_root = f"gs://{data_bucket}/pipeline_root"

    # Initialize Vertex AI
    aip.init(
        project=project_id,
        location=region,
    )

    job = aip.PipelineJob(
        display_name="tsunami-training-pipeline-run",
        template_path="tsunami_training_pipeline.yaml",
        enable_caching=False,
        pipeline_root=pipeline_root,
        parameter_values={
            'project_id': project_id,
            'data_bucket': data_bucket,
            'data_filename': 'earthquake_data_tsunami.csv',
            'model_bucket': model_bucket,
            'temp_bucket': temp_bucket,
            'deployed_model_file': 'deployed_model.pkl',
            'model_candidate_file': 'model_candidate.pkl',
            'cicd_webhook_url': os.getenv('CICD_WEBHOOK_URL', 'none')
        }
    )

    print('🚀 Submitting pipeline to Vertex AI...')
    print(f'Project: {project_id}')
    print(f'Region: {region}')
    print(f'Pipeline Root: {pipeline_root}')
    
    job.run()
    print(f'✅ Pipeline submitted! View at: https://console.cloud.google.com/vertex-ai/pipelines/runs?project={project_id}')


# ==================== MAIN ====================

if __name__ == '__main__':
    compile_pipeline()
    # Uncomment the line below to automatically run the pipeline after compilation
    # run_pipeline()