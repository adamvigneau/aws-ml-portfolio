import boto3
import json
from datetime import datetime
from io import StringIO

with open('mlops_config.json', 'r') as f:
    config = json.load(f)

s3 = boto3.client('s3')

# Create simple test data
test_data = "1,2,3,4,5\n6,7,8,9,10\n"

# Upload to trigger pipeline
s3_key = f"training-data/test-{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"

print(f"Uploading test file to: s3://{config['training_bucket']}/{s3_key}")

s3.put_object(
    Bucket=config['training_bucket'],
    Key=s3_key,
    Body=test_data
)

print("✅ File uploaded!")
print("\nWait 30 seconds, then check Lambda 1 logs:")
print("aws logs tail /aws/lambda/MLOps-TriggerTraining --follow --since 5m")