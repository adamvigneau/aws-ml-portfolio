import boto3
import json

# Load config
with open('mlops_config.json', 'r') as f:
    config = json.load(f)

sagemaker_client = boto3.client('sagemaker')

print("🔍 Looking for models awaiting approval...")

pkgs = sagemaker_client.list_model_packages(
    ModelPackageGroupName='mlops-pipeline-models',
    SortBy='CreationTime',
    SortOrder='Descending'
)

for pkg in pkgs['ModelPackageSummaryList']:
    if pkg['ModelApprovalStatus'] == 'PendingManualApproval':
        model_arn = pkg['ModelPackageArn']
        
        print(f"\n✅ Found model: {model_arn}")
        print("Approving...")
        
        sagemaker_client.update_model_package(
            ModelPackageArn=model_arn,
            ModelApprovalStatus='Approved',
            ApprovalDescription='Approved for production'
        )
        
        print("✅ Model approved!")
        print("Deployment will start automatically in ~1 minute")
        print("Check your email for deployment notification")
        break
else:
    print("No models awaiting approval")