"""
Train 5 XGBoost model variants for Multi-Model Endpoint demo
"""
import boto3
import sagemaker
from sagemaker import image_uris
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
import pandas as pd
import numpy as np
import json
import time

# ============================================================
# CONFIGURATION - Update role with your SageMaker execution role
# ============================================================
role = "arn:aws:iam::854757836160:role/service-role/AmazonSageMaker-ExecutionRole-20251019T120276"

# Configuration
region = boto3.Session().region_name
boto_session = boto3.Session(region_name=region)
sagemaker_session = sagemaker.Session(boto_session=boto_session)

bucket = sagemaker_session.default_bucket()
prefix = 'mme-demo'

print(f"Region: {region}")
print(f"Bucket: {bucket}")
print(f"Prefix: {prefix}")

print("\n" + "="*70)
print("STEP 1: CREATE TRAINING DATASET")
print("="*70)

# Create a classification dataset
np.random.seed(42)
n_samples = 2000

data = {
    'feature_1': np.random.randn(n_samples),
    'feature_2': np.random.randn(n_samples),
    'feature_3': np.random.randn(n_samples),
    'feature_4': np.random.randn(n_samples),
    'feature_5': np.random.randn(n_samples),
}

df = pd.DataFrame(data)
df['target'] = ((df['feature_1'] + df['feature_2'] * 0.5 + np.random.randn(n_samples) * 0.3) > 0).astype(int)

# XGBoost format: target first, no headers
train_df = df[['target', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']]

print(f"\nDataset shape: {train_df.shape}")
print(f"Target distribution:\n{train_df['target'].value_counts()}")

# Save and upload
train_file = 'train.csv'
train_df.to_csv(train_file, index=False, header=False)

train_s3_path = f's3://{bucket}/{prefix}/train/{train_file}'
sagemaker_session.upload_data(train_file, bucket=bucket, key_prefix=f'{prefix}/train')

print(f"\n✅ Training data uploaded to: {train_s3_path}")

print("\n" + "="*70)
print("STEP 2: DEFINE 5 MODEL VARIANTS")
print("="*70)

# 5 different hyperparameter configurations
# Simulating customer-specific or A/B test variants
model_variants = {
    'model_conservative': {
        'num_round': 25,
        'max_depth': 2,
        'eta': 0.3,
        'description': 'Simple, fast, low risk of overfitting'
    },
    'model_balanced': {
        'num_round': 50,
        'max_depth': 4,
        'eta': 0.2,
        'description': 'Balanced performance'
    },
    'model_aggressive': {
        'num_round': 100,
        'max_depth': 6,
        'eta': 0.1,
        'description': 'Complex, potentially more accurate'
    },
    'model_deep': {
        'num_round': 50,
        'max_depth': 8,
        'eta': 0.2,
        'description': 'Deep trees, captures complex patterns'
    },
    'model_fast': {
        'num_round': 30,
        'max_depth': 3,
        'eta': 0.4,
        'description': 'Optimized for speed'
    }
}

print("\nModel variants to train:")
for name, config in model_variants.items():
    print(f"\n   {name}:")
    print(f"      num_round: {config['num_round']}")
    print(f"      max_depth: {config['max_depth']}")
    print(f"      eta: {config['eta']}")
    print(f"      → {config['description']}")

print("\n" + "="*70)
print("STEP 3: TRAIN ALL 5 MODELS")
print("="*70)

# Get XGBoost container
container = image_uris.retrieve('xgboost', region, '1.5-1')
print(f"\nUsing container: {container}")

# Store model artifacts
model_artifacts = {}

for i, (model_name, config) in enumerate(model_variants.items(), 1):
    print(f"\n{'─'*70}")
    print(f"Training model {i}/5: {model_name}")
    print(f"{'─'*70}")
    print(f"   Hyperparameters: num_round={config['num_round']}, "
          f"max_depth={config['max_depth']}, eta={config['eta']}")
    
    # Create estimator
    estimator = Estimator(
        image_uri=container,
        role=role,
        instance_count=1,
        instance_type='ml.m5.large',
        output_path=f's3://{bucket}/{prefix}/models/{model_name}/',
        sagemaker_session=sagemaker_session,
        base_job_name=f'mme-{model_name.replace("_", "-")}'
    )
    
    # Set hyperparameters
    estimator.set_hyperparameters(
        objective='binary:logistic',
        num_round=config['num_round'],
        max_depth=config['max_depth'],
        eta=config['eta'],
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    # Train
    print(f"   Starting training job...")
    start_time = time.time()
    
    estimator.fit(
        {'train': TrainingInput(train_s3_path, content_type='text/csv')},
        wait=True,
        logs=False  # Suppress detailed logs
    )
    
    elapsed = time.time() - start_time
    
    # Store artifact location
    model_artifacts[model_name] = estimator.model_data
    
    print(f"   ✅ Completed in {elapsed:.0f}s")
    print(f"   📦 Artifact: {estimator.model_data}")

print("\n" + "="*70)
print("✅ ALL 5 MODELS TRAINED!")
print("="*70)

print("\nModel artifacts:")
for name, artifact in model_artifacts.items():
    print(f"   {name}: {artifact}")

# Save model artifacts for next step
with open('model_artifacts.json', 'w') as f:
    json.dump(model_artifacts, f, indent=2)

print("\n📝 Model artifacts saved to: model_artifacts.json")
print("\n💡 Next step: python deploy_multi_model_endpoint.py")