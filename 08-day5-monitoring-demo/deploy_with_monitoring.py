"""
Deploy a model with Data Capture enabled for Model Monitor
"""
import boto3
import sagemaker
from sagemaker import image_uris
from sagemaker.model import Model
from sagemaker.model_monitor import DataCaptureConfig
import pandas as pd
import numpy as np
import json
import time

# ============================================================
# CONFIGURATION
# ============================================================
role = "arn:aws:iam::854757836160:role/service-role/AmazonSageMaker-ExecutionRole-20251019T120276"

region = boto3.Session().region_name
sagemaker_session = sagemaker.Session()
sagemaker_client = boto3.client('sagemaker', region_name=region)

bucket = sagemaker_session.default_bucket()
prefix = 'monitoring-demo'

print(f"Region: {region}")
print(f"Bucket: {bucket}")
print(f"Prefix: {prefix}")

print("\n" + "="*70)
print("STEP 1: CREATE AND UPLOAD TRAINING DATA")
print("="*70)

# Create a dataset
np.random.seed(42)
n_samples = 1000

# Features with specific distributions (we'll drift these later)
data = {
    'age': np.random.normal(35, 10, n_samples).clip(18, 80),
    'income': np.random.normal(50000, 15000, n_samples).clip(20000, 150000),
    'credit_score': np.random.normal(700, 50, n_samples).clip(300, 850),
    'loan_amount': np.random.normal(25000, 10000, n_samples).clip(5000, 100000),
    'employment_years': np.random.normal(8, 5, n_samples).clip(0, 40),
}

df = pd.DataFrame(data)

# Target: loan approval (binary)
df['approved'] = (
    (df['credit_score'] > 650) & 
    (df['income'] > 35000) & 
    (df['loan_amount'] < df['income'] * 0.5)
).astype(int)

# XGBoost format
train_df = df[['approved', 'age', 'income', 'credit_score', 'loan_amount', 'employment_years']]

print(f"\nDataset shape: {train_df.shape}")
print(f"Approval rate: {train_df['approved'].mean()*100:.1f}%")
print(f"\nFeature statistics:")
print(df.describe().round(2))

# Save and upload
train_file = 'train.csv'
train_df.to_csv(train_file, index=False, header=False)

from sagemaker.inputs import TrainingInput
train_s3_path = f's3://{bucket}/{prefix}/train/{train_file}'
sagemaker_session.upload_data(train_file, bucket=bucket, key_prefix=f'{prefix}/train')

print(f"\n✅ Training data uploaded to: {train_s3_path}")

# Save baseline data (features only, no target) for Model Monitor
baseline_df = df[['age', 'income', 'credit_score', 'loan_amount', 'employment_years']]
baseline_file = 'baseline.csv'
baseline_df.to_csv(baseline_file, index=False)

baseline_s3_path = f's3://{bucket}/{prefix}/baseline/{baseline_file}'
sagemaker_session.upload_data(baseline_file, bucket=bucket, key_prefix=f'{prefix}/baseline')

print(f"✅ Baseline data uploaded to: {baseline_s3_path}")

print("\n" + "="*70)
print("STEP 2: TRAIN MODEL")
print("="*70)

from sagemaker.estimator import Estimator

container = image_uris.retrieve('xgboost', region, '1.5-1')
print(f"\nContainer: {container}")

estimator = Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    output_path=f's3://{bucket}/{prefix}/model/',
    sagemaker_session=sagemaker_session,
    base_job_name='monitoring-demo'
)

estimator.set_hyperparameters(
    objective='binary:logistic',
    num_round=50,
    max_depth=4,
    eta=0.2
)

print("\nTraining model...")
estimator.fit(
    {'train': TrainingInput(train_s3_path, content_type='text/csv')},
    wait=True,
    logs=False
)

model_artifact = estimator.model_data
print(f"\n✅ Model trained: {model_artifact}")

print("\n" + "="*70)
print("STEP 3: DEPLOY WITH DATA CAPTURE")
print("="*70)

timestamp = int(time.time())
model_name = f'monitoring-demo-model-{timestamp}'
endpoint_config_name = f'monitoring-demo-config-{timestamp}'
endpoint_name = f'monitoring-demo-endpoint-{timestamp}'

# Data capture location
data_capture_prefix = f'{prefix}/data-capture'
data_capture_s3_uri = f's3://{bucket}/{data_capture_prefix}'

print(f"\nData capture location: {data_capture_s3_uri}")

# Create Model
print(f"\n1️⃣  Creating Model: {model_name}")

sagemaker_client.create_model(
    ModelName=model_name,
    PrimaryContainer={
        'Image': container,
        'ModelDataUrl': model_artifact
    },
    ExecutionRoleArn=role
)
print(f"   ✅ Model created")

# Create Endpoint Config WITH Data Capture
print(f"\n2️⃣  Creating Endpoint Config with Data Capture: {endpoint_config_name}")

sagemaker_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[{
        'VariantName': 'AllTraffic',
        'ModelName': model_name,
        'InitialInstanceCount': 1,
        'InstanceType': 'ml.m5.large'
    }],
    DataCaptureConfig={
        'EnableCapture': True,
        'InitialSamplingPercentage': 100,  # Capture 100% of requests
        'DestinationS3Uri': data_capture_s3_uri,
        'CaptureOptions': [
            {'CaptureMode': 'Input'},   # Capture request data
            {'CaptureMode': 'Output'}   # Capture response data
        ],
        'CaptureContentTypeHeader': {
            'CsvContentTypes': ['text/csv'],
            'JsonContentTypes': ['application/json']
        }
    }
)
print(f"   ✅ Endpoint config created with data capture enabled")

# Create Endpoint
print(f"\n3️⃣  Creating Endpoint: {endpoint_name}")
print(f"   This will take 3-5 minutes...\n")

sagemaker_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)

# Wait for endpoint
while True:
    response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
    status = response['EndpointStatus']
    
    if status == 'InService':
        print(f"   ✅ Endpoint is InService!")
        break
    elif status == 'Failed':
        print(f"   ❌ Endpoint failed: {response.get('FailureReason', 'Unknown')}")
        exit(1)
    else:
        print(f"   ⏳ Status: {status}...")
        time.sleep(30)

# Save config for later scripts
config = {
    'endpoint_name': endpoint_name,
    'endpoint_config_name': endpoint_config_name,
    'model_name': model_name,
    'model_artifact': model_artifact,
    'data_capture_s3_uri': data_capture_s3_uri,
    'baseline_s3_path': baseline_s3_path,
    'bucket': bucket,
    'prefix': prefix
}

with open('monitoring_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("\n" + "="*70)
print("✅ ENDPOINT DEPLOYED WITH DATA CAPTURE!")
print("="*70)

print(f"""
Endpoint: {endpoint_name}
Data Capture: ENABLED (100% sampling)
Capture Location: {data_capture_s3_uri}

📝 Config saved to: monitoring_config.json

💡 Next step: python create_baseline.py
""")

print("\n" + "-"*70)
print("WHAT IS DATA CAPTURE?")
print("-"*70)
print("""
Data Capture records all inference requests and responses to S3:

   Request ──→ [Endpoint] ──→ Response
                   │
                   ▼
              S3 Bucket
           (captured data)

This captured data is used by Model Monitor to:
1. Compare against baseline statistics
2. Detect data drift
3. Alert when anomalies are found
""")