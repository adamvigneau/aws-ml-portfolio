"""
Cleanup Week 8 SageMaker Pipeline resources
"""
import boto3
import time

print("🧹 Week 8 Pipeline Cleanup")
print("="*70)

sm = boto3.client('sagemaker')
s3 = boto3.client('s3')

# 1. Delete Pipeline
print("\n1. Deleting SageMaker Pipeline...")
try:
    sm.delete_pipeline(PipelineName='MLOpsPipeline')
    print("   ✅ Pipeline deleted")
except Exception as e:
    print(f"   ⚠️  {e}")

# 2. Check for endpoints (shouldn't exist but check)
print("\n2. Checking for endpoints...")
try:
    sm.describe_endpoint(EndpointName='mlops-production-endpoint')
    print("   Found endpoint, deleting...")
    sm.delete_endpoint(EndpointName='mlops-production-endpoint')
    print("   ✅ Endpoint deleted")
except:
    print("   ⚠️  No endpoint found")

# 3. List and optionally delete models/configs
print("\n3. Cleaning up models and configs...")
try:
    # List models
    models = sm.list_models(NameContains='mlops-model', MaxResults=10)
    for model in models['Models']:
        print(f"   Model: {model['ModelName']}")
        delete = input(f"   Delete {model['ModelName']}? (y/n): ")
        if delete.lower() == 'y':
            sm.delete_model(ModelName=model['ModelName'])
            print(f"   ✅ Deleted")
    
    # List endpoint configs
    configs = sm.list_endpoint_configs(NameContains='mlops', MaxResults=10)
    for config in configs['EndpointConfigs']:
        print(f"   Config: {config['EndpointConfigName']}")
        delete = input(f"   Delete {config['EndpointConfigName']}? (y/n): ")
        if delete.lower() == 'y':
            sm.delete_endpoint_config(EndpointConfigName=config['EndpointConfigName'])
            print(f"   ✅ Deleted")
except Exception as e:
    print(f"   ⚠️  {e}")

# 4. S3 cleanup
print("\n4. S3 Buckets...")
bucket_name = 'sagemaker-us-east-2-854757836160'
print(f"   Bucket: {bucket_name}")
print("   Contains: pipeline scripts, outputs, and logs")

delete_s3 = input(f"   Delete pipeline-related S3 objects? (yes/no): ")
if delete_s3.lower() == 'yes':
    try:
        # Delete pipeline scripts
        objects = s3.list_objects_v2(Bucket=bucket_name, Prefix='pipeline-code/')
        if 'Contents' in objects:
            for obj in objects['Contents']:
                s3.delete_object(Bucket=bucket_name, Key=obj['Key'])
                print(f"   ✅ Deleted {obj['Key']}")
        
        # Delete pipeline outputs
        objects = s3.list_objects_v2(Bucket=bucket_name, Prefix='pipeline-output/')
        if 'Contents' in objects:
            for obj in objects['Contents']:
                s3.delete_object(Bucket=bucket_name, Key=obj['Key'])
                print(f"   ✅ Deleted {obj['Key']}")
    except Exception as e:
        print(f"   ⚠️  {e}")
else:
    print("   ⏭️  Skipped S3 cleanup")

# 5. Model Registry
print("\n5. Model Registry...")
print("   Group: mlops-pipeline-models")
delete_registry = input("   Delete ALL model packages? (yes/no): ")

if delete_registry.lower() == 'yes':
    try:
        packages = sm.list_model_packages(
            ModelPackageGroupName='mlops-pipeline-models',
            MaxResults=100
        )
        
        for pkg in packages['ModelPackageSummaryList']:
            sm.delete_model_package(ModelPackageName=pkg['ModelPackageArn'])
            version = pkg['ModelPackageArn'].split('/')[-1]
            print(f"   ✅ Deleted version {version}")
        
        print("   Note: Model Package Group will remain (contains history)")
    except Exception as e:
        print(f"   ⚠️  {e}")
else:
    print("   ⏭️  Skipped")

print("\n" + "="*70)
print("✅ CLEANUP COMPLETE!")
print("="*70)
print("\nRemaining resources (if any):")
print("  - mlops-pipeline-models group (for history)")
print("  - IAM roles (shared with Week 7)")
print("  - S3 bucket (shared)")
print("\n💰 No ongoing costs!")
