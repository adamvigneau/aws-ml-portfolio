"""
Cleanup SageMaker Feature Store Resources
"""
import boto3
import time

# Configuration
region = boto3.Session().region_name
sagemaker_client = boto3.client('sagemaker', region_name=region)

print(f"Region: {region}")

# Feature group names
feature_groups = [
    "users-feature-group",
    "products-feature-group"
]

print("\n" + "="*70)
print("CLEANING UP FEATURE STORE RESOURCES")
print("="*70)

# ============================================================
# Delete Feature Groups
# ============================================================
for fg_name in feature_groups:
    print(f"\n🗑️  Deleting {fg_name}...")
    
    try:
        sagemaker_client.delete_feature_group(
            FeatureGroupName=fg_name
        )
        print(f"   ✅ Delete initiated: {fg_name}")
        
        # Wait for deletion
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                response = sagemaker_client.describe_feature_group(
                    FeatureGroupName=fg_name
                )
                status = response.get("FeatureGroupStatus")
                print(f"   ⏳ Status: {status} (attempt {attempt + 1}/{max_attempts})")
                time.sleep(5)
            except sagemaker_client.exceptions.ResourceNotFound:
                print(f"   ✅ Deleted: {fg_name}")
                break
            except Exception as e:
                if "ResourceNotFound" in str(e) or "does not exist" in str(e):
                    print(f"   ✅ Deleted: {fg_name}")
                    break
                else:
                    print(f"   ⚠️  Error checking status: {e}")
                    time.sleep(5)
                    
    except sagemaker_client.exceptions.ResourceNotFound:
        print(f"   ⚠️  Feature group not found (already deleted?)")
    except Exception as e:
        print(f"   ❌ Error: {e}")

# ============================================================
# Summary
# ============================================================
print("\n" + "="*70)
print("✅ CLEANUP COMPLETE!")
print("="*70)

print("""
📝 What was deleted:
   - Feature groups (online store stops immediately)
   - Glue table references

📝 What remains (optional manual cleanup):
   - S3 data in s3://<bucket>/feature-store/
   - Athena query results in s3://<bucket>/athena-results/

To delete S3 data manually:
   aws s3 rm s3://sagemaker-us-east-2-854757836160/feature-store/ --recursive
   aws s3 rm s3://sagemaker-us-east-2-854757836160/athena-results/ --recursive

💡 The online store (DynamoDB) stops billing immediately after deletion.
   The S3 data is minimal cost but can be removed if desired.
""")
