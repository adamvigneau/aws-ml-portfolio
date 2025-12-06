"""
Create a baseline for Model Monitor to compare against
"""
import boto3
import sagemaker
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat
import json
import time

# ============================================================
# CONFIGURATION
# ============================================================
role = "arn:aws:iam::854757836160:role/service-role/AmazonSageMaker-ExecutionRole-20251019T120276"

region = boto3.Session().region_name
sagemaker_session = sagemaker.Session()
sagemaker_client = boto3.client('sagemaker', region_name=region)

print(f"Region: {region}")

# Load config from deployment
with open('monitoring_config.json', 'r') as f:
    config = json.load(f)

bucket = config['bucket']
prefix = config['prefix']
baseline_s3_path = config['baseline_s3_path']

print(f"Bucket: {bucket}")
print(f"Baseline data: {baseline_s3_path}")

print("\n" + "="*70)
print("CREATING BASELINE WITH MODEL MONITOR")
print("="*70)

print("""
What is a baseline?

A baseline captures the statistical properties of your training data:
- Mean, std, min, max for each feature
- Data types and distributions
- Expected ranges

Model Monitor compares live traffic against this baseline to detect drift.
""")

# Create DefaultModelMonitor
monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.t3.large',  # More memory
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600,
    sagemaker_session=sagemaker_session
)

# Baseline output location
baseline_results_uri = f's3://{bucket}/{prefix}/baseline-results'

print(f"\n1️⃣  Starting baseline job...")
print(f"   Input: {baseline_s3_path}")
print(f"   Output: {baseline_results_uri}")
print(f"   This will take 3-5 minutes...\n")

# Suggest baseline (analyze the training data)
monitor.suggest_baseline(
    baseline_dataset=baseline_s3_path,
    dataset_format=DatasetFormat.csv(header=True),
    output_s3_uri=baseline_results_uri,
    wait=True,
    logs=False
)

print(f"   ✅ Baseline job completed!")

# Get the baseline results
baseline_job = monitor.latest_baselining_job
print(f"\n2️⃣  Baseline job name: {baseline_job.job_name}")

print("\n" + "="*70)
print("BASELINE STATISTICS")
print("="*70)

# Download and display baseline statistics
s3_client = boto3.client('s3')

# Get statistics file
stats_key = f"{prefix}/baseline-results/statistics.json"
try:
    response = s3_client.get_object(Bucket=bucket, Key=stats_key)
    statistics = json.loads(response['Body'].read().decode('utf-8'))
    
    print("\nFeature Statistics (what Model Monitor considers 'normal'):\n")
    
    print(f"{'Feature':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("-" * 70)
    
    for feature in statistics.get('features', []):
        name = feature.get('name', 'unknown')
        numerical = feature.get('numerical_statistics', {})
        mean = numerical.get('mean', 'N/A')
        std = numerical.get('std', 'N/A')
        min_val = numerical.get('min', 'N/A')
        max_val = numerical.get('max', 'N/A')
        
        if isinstance(mean, (int, float)):
            print(f"{name:<20} {mean:>12.2f} {std:>12.2f} {min_val:>12.2f} {max_val:>12.2f}")
        else:
            print(f"{name:<20} {str(mean):>12} {str(std):>12} {str(min_val):>12} {str(max_val):>12}")
            
except Exception as e:
    print(f"   Could not load statistics: {e}")
    print(f"   Check S3: s3://{bucket}/{stats_key}")

# Get constraints file
print("\n" + "-"*70)
print("CONSTRAINTS (Drift Detection Rules)")
print("-"*70)

constraints_key = f"{prefix}/baseline-results/constraints.json"
try:
    response = s3_client.get_object(Bucket=bucket, Key=constraints_key)
    constraints = json.loads(response['Body'].read().decode('utf-8'))
    
    print("\nModel Monitor will alert if these constraints are violated:\n")
    
    monitoring_config = constraints.get('monitoring_config', {})
    
    # Distribution constraints
    dist_constraints = monitoring_config.get('distribution_constraints', {})
    if dist_constraints:
        print(f"   Distribution comparison: {dist_constraints.get('comparison_type', 'N/A')}")
        print(f"   Threshold: {dist_constraints.get('threshold', 'N/A')}")
    
    # Data type constraints
    print(f"\n   Data type monitoring: Enabled")
    print(f"   Completeness monitoring: Enabled")
    
except Exception as e:
    print(f"   Could not load constraints: {e}")

# Update config with baseline info
config['baseline_results_uri'] = baseline_results_uri
config['baseline_statistics'] = f's3://{bucket}/{stats_key}'
config['baseline_constraints'] = f's3://{bucket}/{constraints_key}'

with open('monitoring_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("\n" + "="*70)
print("✅ BASELINE CREATED!")
print("="*70)

print(f"""
Baseline Results: {baseline_results_uri}
- statistics.json: Feature distributions
- constraints.json: Drift detection rules

📝 Config updated: monitoring_config.json

💡 Next step: python send_traffic_and_drift.py

This will:
1. Send normal traffic to establish a pattern
2. Send drifted traffic to trigger alerts
3. Check for drift violations
""")