"""
SageMaker Pipeline Definition - Local Execution
"""
import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingOutput
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.properties import PropertyFile
from sagemaker.processing import ProcessingInput
import json
import os
from sagemaker.processing import ScriptProcessor


# Get configuration
region = boto3.Session().region_name
sagemaker_session = sagemaker.Session()
bucket = sagemaker_session.default_bucket()

# Load role from Lambda pipeline config
config_path = os.path.expanduser('~/mlops-pipeline/mlops_config.json')
if os.path.exists(config_path):
    with open(config_path) as f:
        config = json.load(f)
        role = config['sagemaker_role_arn']
else:
    # Fallback: specify manually
    account_id = boto3.client('sts').get_caller_identity()['Account']
    role = f"arn:aws:iam::{account_id}:role/MLOpsSageMakerRole"

print(f"Region: {region}")
print(f"Bucket: {bucket}")
print(f"Role: {role}")

# Define pipeline parameters
processing_instance_type = ParameterString(
    name="ProcessingInstanceType",
    default_value="ml.m5.xlarge"
)

training_instance_type = ParameterString(
    name="TrainingInstanceType",
    default_value="ml.m5.xlarge"
)

accuracy_threshold = ParameterFloat(
    name="AccuracyThreshold",
    default_value=0.75
)

# Upload scripts to S3
print("\nUploading scripts to S3...")
preprocess_s3 = sagemaker_session.upload_data(
    path='preprocess.py',
    bucket=bucket,
    key_prefix='pipeline-code'
)
evaluate_s3 = sagemaker_session.upload_data(
    path='evaluate.py',
    bucket=bucket,
    key_prefix='pipeline-code'
)
print(f"Scripts uploaded to s3://{bucket}/pipeline-code/")

# Step 1: Processing
print("\nCreating processing step...")
sklearn_processor = SKLearnProcessor(
    framework_version="1.2-1",
    instance_type=processing_instance_type,
    instance_count=1,
    base_job_name="pipeline-preprocess",
    role=role,
    sagemaker_session=sagemaker_session
)

step_process = ProcessingStep(
    name="PreprocessData",
    processor=sklearn_processor,
    outputs=[
        ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
        ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation")
    ],
    code=preprocess_s3
)

# Step 2: Training
print("Creating training step...")
image_uri = sagemaker.image_uris.retrieve(
    framework="xgboost",
    region=region,
    version="1.5-1",
    py_version="py3"
)

xgb_estimator = Estimator(
    image_uri=image_uri,
    instance_type=training_instance_type,
    instance_count=1,
    output_path=f"s3://{bucket}/pipeline-output",
    base_job_name="pipeline-train",
    role=role,
    sagemaker_session=sagemaker_session,
    hyperparameters={
        "objective": "binary:logistic",
        "num_round": "100",
        "max_depth": "5",
        "eta": "0.2",
        "subsample": "0.8"
    }
)

step_train = TrainingStep(
    name="TrainModel",
    estimator=xgb_estimator,
    inputs={
        "train": TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            content_type="text/csv"
        ),
        "validation": TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
            content_type="text/csv"
        )
    }
)

# Step 3: Evaluation with XGBoost container
print("3️⃣  Creating Evaluation Step...")

xgboost_image_uri = sagemaker.image_uris.retrieve(
    framework="xgboost",
    region=region,
    version="1.5-1"
)

eval_processor = ScriptProcessor(
    image_uri=xgboost_image_uri,
    instance_type=processing_instance_type,
    instance_count=1,
    base_job_name="pipeline-eval",
    role=role,
    sagemaker_session=sagemaker_session,
    command=["python3"]
)

evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json"
)

step_eval = ProcessingStep(
    name="EvaluateModel",
    processor=eval_processor,
    inputs=[
        ProcessingInput(
            source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
        ProcessingInput(
            source=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
            destination="/opt/ml/processing/test"
        )
    ],
    outputs=[
        ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")
    ],
    code=f"s3://{bucket}/pipeline-code/evaluate.py",
    property_files=[evaluation_report]
)

# Step 4: Register Model (with metrics)
print("Creating register model step...")
model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri="{}/evaluation.json".format(
            step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
        ),
        content_type="application/json"
    )
)

step_register = RegisterModel(
    name="RegisterModel",
    estimator=xgb_estimator,
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    inference_instances=["ml.t2.medium", "ml.m5.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name="mlops-pipeline-models",
    approval_status="PendingManualApproval",
    model_metrics=model_metrics
)

# Step 5: Condition (only register if accuracy meets threshold)
print("Creating condition step...")
cond_gte = ConditionGreaterThanOrEqualTo(
    left=JsonGet(
        step_name=step_eval.name,
        property_file=evaluation_report,
        json_path="accuracy"
    ),
    right=accuracy_threshold
)

step_cond = ConditionStep(
    name="CheckAccuracy",
    conditions=[cond_gte],
    if_steps=[step_register],
    else_steps=[]
)

# Create Pipeline
print("Creating pipeline...")
pipeline = Pipeline(
    name="MLOpsPipeline",
    parameters=[
        processing_instance_type,
        training_instance_type,
        accuracy_threshold
    ],
    steps=[step_process, step_train, step_eval, step_cond],
    sagemaker_session=sagemaker_session
)

print("\n" + "="*70)
print("PIPELINE CREATED")
print("="*70)
print(f"Name: {pipeline.name}")
print(f"Steps: {len(pipeline.steps)}")
print(f"Parameters: {len(pipeline.parameters)}")

# Upsert (create or update) pipeline
print("\nUpserting pipeline to SageMaker...")
response = pipeline.upsert(role_arn=role)

print(f"\n✅ Pipeline ready!")
print(f"Pipeline ARN: {response['PipelineArn']}")