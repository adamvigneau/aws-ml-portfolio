import boto3
import json

# Load config
with open('mlops_config.json', 'r') as f:
    config = json.load(f)

sagemaker_client = boto3.client('sagemaker')

print("🔍 Looking for models awaiting approval...")

try:
    pkgs = sagemaker_client.list_model_packages(
        ModelPackageGroupName='mlops-pipeline-models',
        SortBy='CreationTime',
        SortOrder='Descending'
    )

    pending_models = []
    for pkg in pkgs['ModelPackageSummaryList']:
        if pkg['ModelApprovalStatus'] == 'PendingManualApproval':
            pending_models.append(pkg)

    if not pending_models:
        print("\n❌ No models awaiting approval")
        print("\nCurrent models:")
        for pkg in pkgs['ModelPackageSummaryList'][:5]:
            name = pkg['ModelPackageArn'].split('/')[-1]
            print(f"  {name}: {pkg['ModelApprovalStatus']}")
    else:
        print(f"\n✅ Found {len(pending_models)} model(s) awaiting approval:\n")
        
        for i, pkg in enumerate(pending_models, 1):
            model_arn = pkg['ModelPackageArn']
            name = model_arn.split('/')[-1]
            print(f"{i}. {name}")
            print(f"   ARN: {model_arn}")
            print(f"   Created: {pkg['CreationTime']}")
            print()
        
        # Approve the most recent one
        if len(pending_models) == 1:
            approve = input("Approve this model? (y/n): ").strip().lower()
        else:
            approve = input(f"Approve model #1 (most recent)? (y/n): ").strip().lower()
        
        if approve == 'y':
            model_arn = pending_models[0]['ModelPackageArn']
            
            print(f"\nApproving: {model_arn}")
            
            sagemaker_client.update_model_package(
                ModelPackageArn=model_arn,
                ModelApprovalStatus='Approved',
                ApprovalDescription='Approved for production deployment'
            )
            
            print("✅ Model approved!")
            print("\n🚀 Deployment will start automatically in ~1 minute")
            print("📧 Check your email for deployment notification")
        else:
            print("\nApproval cancelled")

except Exception as e:
    print(f"\n❌ Error: {e}")