"""
Deploy 5 models to a single Multi-Model Endpoint (MME)
"""
import boto3
import sagemaker
from sagemaker import image_uris
import json
import time
import tarfile
import os

# ============================================================
# CONFIGURATION
# ============================================================
role = "arn:aws:iam::854757836160:role/service-role/AmazonSageMaker-ExecutionRole-20251019T120276"

region = boto3.Session().region_name
sagemaker_session = sagemaker.Session()
sagemaker_client = boto3.client('sagemaker', region_name=region)
s3_client = boto3.client('s3', region_name=region)

bucket = sagemaker_session.default_bucket()
prefix = 'mme-demo'

print(f"Region: {region}")
print(f"Bucket: {bucket}")

# Load model artifacts from training
with open('model_artifacts.json', 'r') as f:
    model_artifacts = json.load(f)

print(f"\nLoaded {len(model_artifacts)} model artifacts")

print("\n" + "="*70)
print("STEP 1: COPY MODELS TO MME LOCATION")
print("="*70)

# MME requires all models in a single S3 prefix
mme_model_prefix = f'{prefix}/mme-models'
print(f"\nMME model location: s3://{bucket}/{mme_model_prefix}/")

# Copy each model to the MME location with a simple name
for model_name, artifact_path in model_artifacts.items():
    # Parse source location
    source_path = artifact_path.replace(f's3://{bucket}/', '')
    
    # Destination: simple name like "model_conservative.tar.gz"
    dest_key = f'{mme_model_prefix}/{model_name}.tar.gz'
    
    print(f"\n   Copying {model_name}...")
    print(f"   From: {source_path}")
    print(f"   To: {dest_key}")
    
    s3_client.copy_object(
        Bucket=bucket,
        CopySource={'Bucket': bucket, 'Key': source_path},
        Key=dest_key
    )
    print(f"   ✅ Copied")

print("\n" + "="*70)
print("STEP 2: CREATE MULTI-MODEL ENDPOINT")
print("="*70)

timestamp = int(time.time())
model_name = f'mme-demo-model-{timestamp}'
endpoint_config_name = f'mme-demo-config-{timestamp}'
endpoint_name = f'mme-demo-endpoint-{timestamp}'

# Get XGBoost container (must support MME)
container = image_uris.retrieve('xgboost', region, '1.5-1')
print(f"\nContainer: {container}")

# Create Model (pointing to the MME prefix, not a single model)
print(f"\n1️⃣  Creating MME Model: {model_name}")

sagemaker_client.create_model(
    ModelName=model_name,
    PrimaryContainer={
        'Image': container,
        'Mode': 'MultiModel',  # This makes it an MME!
        'ModelDataUrl': f's3://{bucket}/{mme_model_prefix}/'
    },
    ExecutionRoleArn=role
)
print(f"   ✅ Model created")

# Create Endpoint Config
print(f"\n2️⃣  Creating Endpoint Config: {endpoint_config_name}")

sagemaker_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[{
        'VariantName': 'AllModels',
        'ModelName': model_name,
        'InitialInstanceCount': 1,
        'InstanceType': 'ml.m5.large'
    }]
)
print(f"   ✅ Endpoint config created")

# Create Endpoint
print(f"\n3️⃣  Creating Endpoint: {endpoint_name}")
print(f"   This will take 3-5 minutes...\n")

sagemaker_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)

# Wait for endpoint to be ready
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

# Save endpoint info for testing and cleanup
endpoint_info = {
    'endpoint_name': endpoint_name,
    'endpoint_config_name': endpoint_config_name,
    'model_name': model_name,
    'models': list(model_artifacts.keys())
}

with open('endpoint_info.json', 'w') as f:
    json.dump(endpoint_info, f, indent=2)

print("\n" + "="*70)
print("✅ MULTI-MODEL ENDPOINT DEPLOYED!")
print("="*70)

print(f"""
Endpoint: {endpoint_name}
Models available:
   - model_conservative.tar.gz
   - model_balanced.tar.gz
   - model_aggressive.tar.gz
   - model_deep.tar.gz
   - model_fast.tar.gz

📝 Endpoint info saved to: endpoint_info.json

💡 Next step: python test_mme.py
""")

print("\n" + "-"*70)
print("COST COMPARISON")
print("-"*70)
print(f"""
┌─────────────────────────────────────────────────────────────┐
│  Deployment Strategy      │  Endpoints  │  Cost/hour       │
├─────────────────────────────────────────────────────────────┤
│  5 Separate Endpoints     │     5       │  5 × $0.10 = $0.50│
│  1 Multi-Model Endpoint   │     1       │  1 × $0.10 = $0.10│
├─────────────────────────────────────────────────────────────┤
│  💰 SAVINGS               │    -4       │  80% cheaper!     │
└─────────────────────────────────────────────────────────────┘

(Costs based on ml.m5.large at ~$0.10/hour)
""")