"""
Create SageMaker Feature Groups using boto3 client directly
"""
import boto3
import sagemaker
import time
import json

# Configuration
region = boto3.Session().region_name
boto_session = boto3.Session(region_name=region)
sagemaker_client = boto3.client('sagemaker', region_name=region)

sagemaker_session = sagemaker.Session(
    boto_session=boto_session,
    sagemaker_client=sagemaker_client
)

# Get role
role = None
try:
    with open('../mlops-pipeline/mlops_config.json') as f:
        config = json.load(f)
        role = config['sagemaker_role_arn']
except:
    role = sagemaker.get_execution_role()

print(f"Region: {region}")
print(f"Role: {role}")

# S3 bucket for offline store
bucket = sagemaker_session.default_bucket()
prefix = 'feature-store'

print(f"S3 Bucket: {bucket}")
print(f"Prefix: {prefix}")

# Helper function to wait for feature group creation
def wait_for_feature_group(fg_name, max_attempts=20):
    for attempt in range(max_attempts):
        try:
            response = sagemaker_client.describe_feature_group(
                FeatureGroupName=fg_name
            )
            status = response.get("FeatureGroupStatus")
            
            if status == "Created":
                print(f"   ✅ Status: {status}")
                return True
            elif status == "CreateFailed":
                print(f"   ❌ Status: {status}")
                print(f"   Failure reason: {response.get('FailureReason', 'Unknown')}")
                return False
            else:
                print(f"   ⏳ Status: {status} (attempt {attempt + 1}/{max_attempts})")
                time.sleep(5)
        except Exception as e:
            if attempt == 0:
                print(f"   ⏳ Waiting for feature group to be available...")
            time.sleep(5)
    return False

print("\n" + "="*70)
print("CREATING FEATURE GROUPS")
print("="*70)

# ============================================================
# Feature Group 1: User Features
# ============================================================
print("\n1️⃣  Creating User Feature Group...")

user_feature_group_name = "users-feature-group"

try:
    sagemaker_client.create_feature_group(
        FeatureGroupName=user_feature_group_name,
        RecordIdentifierFeatureName="user_id",
        EventTimeFeatureName="event_time",
        FeatureDefinitions=[
            {"FeatureName": "user_id", "FeatureType": "String"},
            {"FeatureName": "age", "FeatureType": "Integral"},
            {"FeatureName": "membership_tier", "FeatureType": "String"},
            {"FeatureName": "total_purchases", "FeatureType": "Integral"},
            {"FeatureName": "avg_order_value", "FeatureType": "Fractional"},
            {"FeatureName": "event_time", "FeatureType": "Fractional"}
        ],
        OnlineStoreConfig={"EnableOnlineStore": True},
        OfflineStoreConfig={
            "S3StorageConfig": {
                "S3Uri": f"s3://{bucket}/{prefix}"
            }
        },
        RoleArn=role
    )
    
    print(f"   ✅ Created: {user_feature_group_name}")
    print(f"   Record ID: user_id")
    print(f"   Features: 6")
    
except Exception as e:
    if "ResourceInUse" in str(e) or "already exists" in str(e):
        print(f"   ⚠️  Feature group already exists")
    else:
        print(f"   ❌ Error: {e}")

print("   Waiting for feature group to be created...")
time.sleep(2)
wait_for_feature_group(user_feature_group_name)

# ============================================================
# Feature Group 2: Product Features
# ============================================================
print("\n2️⃣  Creating Product Feature Group...")

product_feature_group_name = "products-feature-group"

try:
    sagemaker_client.create_feature_group(
        FeatureGroupName=product_feature_group_name,
        RecordIdentifierFeatureName="product_id",
        EventTimeFeatureName="event_time",
        FeatureDefinitions=[
            {"FeatureName": "product_id", "FeatureType": "String"},
            {"FeatureName": "category", "FeatureType": "String"},
            {"FeatureName": "price", "FeatureType": "Fractional"},
            {"FeatureName": "avg_rating", "FeatureType": "Fractional"},
            {"FeatureName": "stock_level", "FeatureType": "Integral"},
            {"FeatureName": "event_time", "FeatureType": "Fractional"}
        ],
        OnlineStoreConfig={"EnableOnlineStore": True},
        OfflineStoreConfig={
            "S3StorageConfig": {
                "S3Uri": f"s3://{bucket}/{prefix}"
            }
        },
        RoleArn=role
    )
    
    print(f"   ✅ Created: {product_feature_group_name}")
    print(f"   Record ID: product_id")
    print(f"   Features: 6")
    
except Exception as e:
    if "ResourceInUse" in str(e) or "already exists" in str(e):
        print(f"   ⚠️  Feature group already exists")
    else:
        print(f"   ❌ Error: {e}")

print("   Waiting for feature group to be created...")
time.sleep(2)
wait_for_feature_group(product_feature_group_name)

# ============================================================
# Summary
# ============================================================
print("\n" + "="*70)
print("✅ FEATURE GROUPS CREATED!")
print("="*70)
print(f"\nFeature Groups:")
print(f"  1. {user_feature_group_name}")
print(f"  2. {product_feature_group_name}")
print(f"\nOffline Store: s3://{bucket}/{prefix}/")
print(f"Online Store: Enabled (DynamoDB-backed)")
print("\n💡 Tip: View in AWS Console:")
print(f"https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/feature-store")
