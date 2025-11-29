import boto3
import pandas as pd
from io import StringIO
from datetime import datetime
import time
import json

# Load config
with open('mlops_config.json', 'r') as f:
    config = json.load(f)

s3 = boto3.client('s3')
sagemaker_client = boto3.client('sagemaker')

print("🚀 Testing MLOps Pipeline!")
print("="*70)

# Create test data
print("\n1. Creating test training data...")
data = pd.DataFrame({
    'Survived': [0, 1, 1, 0, 1] * 100,
    'Pclass': [3, 1, 3, 1, 3] * 100,
    'Sex': [0, 1, 1, 0, 1] * 100,
    'Age': [22, 38, 26, 35, 35] * 100,
    'SibSp': [1, 1, 0, 1, 0] * 100,
    'Parch': [0, 0, 0, 0, 0] * 100,
    'Fare': [7.25, 71.28, 7.92, 53.10, 8.05] * 100,
    'Embarked': [0, 1, 0, 0, 0] * 100
})

csv_buffer = StringIO()
data.to_csv(csv_buffer, index=False, header=False)

# Upload to S3 (THIS TRIGGERS THE PIPELINE!)
s3_key = f"training-data/train-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
s3.put_object(
    Bucket=config['training_bucket'],
    Key=s3_key,
    Body=csv_buffer.getvalue()
)

print(f"✅ Data uploaded to: s3://{config['training_bucket']}/{s3_key}")
print("\n🎬 PIPELINE TRIGGERED!")
print("="*70)

print("\nWhat happens next:")
print("  1. Training job starts (5-10 min)")
print("  2. Email sent when training completes")
print("  3. Model registered (status: PendingManualApproval)")
print("  4. YOU approve model in console")
print("  5. Deployment happens automatically")
print("  6. Email sent when deployed")

print("\nMonitoring pipeline status...")
print("(Press Ctrl+C to stop monitoring)\n")

def check_status():
    print("\n" + "-"*70)
    print(f"Status check: {datetime.now().strftime('%H:%M:%S')}")
    print("-"*70)
    
    # Training jobs
    jobs = sagemaker_client.list_training_jobs(
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=3
    )
    print("\n📊 Training Jobs:")
    for job in jobs['TrainingJobSummaries']:
        print(f"  {job['TrainingJobName']}: {job['TrainingJobStatus']}")
    
    # Model packages
    try:
        pkgs = sagemaker_client.list_model_packages(
            ModelPackageGroupName='mlops-pipeline-models',
            SortBy='CreationTime',
            SortOrder='Descending',
            MaxResults=3
        )
        print("\n📦 Model Registry:")
        for pkg in pkgs['ModelPackageSummaryList']:
            name = pkg['ModelPackageArn'].split('/')[-1]
            print(f"  {name}: {pkg['ModelApprovalStatus']}")
    except:
        print("\n📦 Model Registry: No models yet")
    
    # Endpoints
    print("\n🌐 Endpoints:")
    try:
        ep = sagemaker_client.describe_endpoint(EndpointName='mlops-production-endpoint')
        print(f"  mlops-production-endpoint: {ep['EndpointStatus']}")
    except:
        print("  No endpoint yet")

try:
    while True:
        check_status()
        time.sleep(120)  # Check every 2 minutes
except KeyboardInterrupt:
    print("\n\nMonitoring stopped.")
    print("\nTo approve model later, run: python mlops_approve.py")
