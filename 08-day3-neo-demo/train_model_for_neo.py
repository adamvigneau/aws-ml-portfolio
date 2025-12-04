"""
Train an XGBoost model for SageMaker Neo compilation
"""
import boto3
import sagemaker
from sagemaker import image_uris
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
import pandas as pd
import numpy as np
import time

# Configuration
region = boto3.Session().region_name
boto_session = boto3.Session(region_name=region)
sagemaker_session = sagemaker.Session(boto_session=boto_session)

bucket = sagemaker_session.default_bucket()
prefix = 'neo-demo'
role = "arn:aws:iam::854757836160:role/service-role/AmazonSageMaker-ExecutionRole-20251019T120276"


print(f"Region: {region}")
print(f"Bucket: {bucket}")
print(f"Prefix: {prefix}")

print("\n" + "="*70)
print("STEP 1: CREATE SAMPLE DATASET")
print("="*70)

# Create a simple classification dataset
np.random.seed(42)
n_samples = 1000

# Generate features
data = {
    'feature_1': np.random.randn(n_samples),
    'feature_2': np.random.randn(n_samples),
    'feature_3': np.random.randn(n_samples),
    'feature_4': np.random.randn(n_samples),
    'feature_5': np.random.randn(n_samples),
}

df = pd.DataFrame(data)

# Create target (binary classification based on features)
df['target'] = ((df['feature_1'] + df['feature_2'] * 0.5 + np.random.randn(n_samples) * 0.3) > 0).astype(int)

# XGBoost requires target in first column, no headers
train_df = df[['target', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']]

print(f"\nDataset shape: {train_df.shape}")
print(f"Target distribution:\n{train_df['target'].value_counts()}")

# Save locally and upload to S3
train_file = 'train.csv'
train_df.to_csv(train_file, index=False, header=False)

train_s3_path = f's3://{bucket}/{prefix}/train/{train_file}'
sagemaker_session.upload_data(train_file, bucket=bucket, key_prefix=f'{prefix}/train')

print(f"\n✅ Training data uploaded to: {train_s3_path}")

print("\n" + "="*70)
print("STEP 2: TRAIN XGBOOST MODEL")
print("="*70)

# Get XGBoost container
container = image_uris.retrieve('xgboost', region, '1.5-1')
print(f"\nUsing container: {container}")

# Create estimator
xgb_estimator = Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    output_path=f's3://{bucket}/{prefix}/output/',
    sagemaker_session=sagemaker_session,
    base_job_name='neo-xgboost'
)

# Set hyperparameters
xgb_estimator.set_hyperparameters(
    objective='binary:logistic',
    num_round=50,
    max_depth=4,
    eta=0.2,
    subsample=0.8,
    colsample_bytree=0.8
)

print("\nStarting training job...")
print("This will take 3-5 minutes...\n")

# Train
xgb_estimator.fit(
    {'train': TrainingInput(train_s3_path, content_type='text/csv')},
    wait=True
)

# Get model artifact location
model_artifact = xgb_estimator.model_data

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print(f"\nModel artifact: {model_artifact}")
print(f"\nTraining job name: {xgb_estimator.latest_training_job.name}")

# Save model path for Neo compilation
with open('model_artifact_path.txt', 'w') as f:
    f.write(model_artifact)

print("\n📝 Model artifact path saved to: model_artifact_path.txt")
print("\n💡 Next step: Compile this model with SageMaker Neo")