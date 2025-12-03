"""
Ingest sample data into SageMaker Feature Groups
"""
import boto3
import sagemaker
import pandas as pd
import time
from sagemaker.feature_store.feature_group import FeatureGroup

# Configuration
region = boto3.Session().region_name
boto_session = boto3.Session(region_name=region)
sagemaker_client = boto3.client('sagemaker', region_name=region)
featurestore_runtime = boto3.client('sagemaker-featurestore-runtime', region_name=region)

sagemaker_session = sagemaker.Session(
    boto_session=boto_session,
    sagemaker_client=sagemaker_client
)

print(f"Region: {region}")

# Feature group names
user_feature_group_name = "users-feature-group"
product_feature_group_name = "products-feature-group"

print("\n" + "="*70)
print("CREATING SAMPLE DATA")
print("="*70)

# ============================================================
# Sample User Data
# ============================================================
print("\n1️⃣  Creating sample user data...")

current_time = time.time()

users_data = pd.DataFrame({
    "user_id": ["user_001", "user_002", "user_003", "user_004", "user_005"],
    "age": [28, 35, 42, 23, 51],
    "membership_tier": ["gold", "silver", "platinum", "bronze", "gold"],
    "total_purchases": [47, 12, 156, 3, 89],
    "avg_order_value": [125.50, 45.00, 312.75, 28.99, 178.25],
    "event_time": [current_time] * 5
})

print(users_data.to_string(index=False))

# ============================================================
# Sample Product Data
# ============================================================
print("\n2️⃣  Creating sample product data...")

products_data = pd.DataFrame({
    "product_id": ["prod_001", "prod_002", "prod_003", "prod_004", "prod_005"],
    "category": ["electronics", "clothing", "home", "electronics", "sports"],
    "price": [299.99, 49.95, 89.00, 599.99, 34.50],
    "avg_rating": [4.5, 4.2, 3.8, 4.9, 4.0],
    "stock_level": [150, 500, 75, 25, 200],
    "event_time": [current_time] * 5
})

print(products_data.to_string(index=False))

print("\n" + "="*70)
print("INGESTING DATA INTO FEATURE GROUPS")
print("="*70)

# ============================================================
# Ingest User Features
# ============================================================
print("\n3️⃣  Ingesting user features...")

user_success = 0
user_failed = 0

for idx, row in users_data.iterrows():
    try:
        record = [
            {"FeatureName": "user_id", "ValueAsString": str(row["user_id"])},
            {"FeatureName": "age", "ValueAsString": str(int(row["age"]))},
            {"FeatureName": "membership_tier", "ValueAsString": str(row["membership_tier"])},
            {"FeatureName": "total_purchases", "ValueAsString": str(int(row["total_purchases"]))},
            {"FeatureName": "avg_order_value", "ValueAsString": str(float(row["avg_order_value"]))},
            {"FeatureName": "event_time", "ValueAsString": str(float(row["event_time"]))}
        ]
        
        featurestore_runtime.put_record(
            FeatureGroupName=user_feature_group_name,
            Record=record
        )
        
        user_success += 1
        print(f"   ✅ Ingested: {row['user_id']}")
        
    except Exception as e:
        user_failed += 1
        print(f"   ❌ Failed: {row['user_id']} - {e}")

print(f"\n   Summary: {user_success} succeeded, {user_failed} failed")

# ============================================================
# Ingest Product Features
# ============================================================
print("\n4️⃣  Ingesting product features...")

product_success = 0
product_failed = 0

for idx, row in products_data.iterrows():
    try:
        record = [
            {"FeatureName": "product_id", "ValueAsString": str(row["product_id"])},
            {"FeatureName": "category", "ValueAsString": str(row["category"])},
            {"FeatureName": "price", "ValueAsString": str(float(row["price"]))},
            {"FeatureName": "avg_rating", "ValueAsString": str(float(row["avg_rating"]))},
            {"FeatureName": "stock_level", "ValueAsString": str(int(row["stock_level"]))},
            {"FeatureName": "event_time", "ValueAsString": str(float(row["event_time"]))}
        ]
        
        featurestore_runtime.put_record(
            FeatureGroupName=product_feature_group_name,
            Record=record
        )
        
        product_success += 1
        print(f"   ✅ Ingested: {row['product_id']}")
        
    except Exception as e:
        product_failed += 1
        print(f"   ❌ Failed: {row['product_id']} - {e}")

print(f"\n   Summary: {product_success} succeeded, {product_failed} failed")

# ============================================================
# Verify: Read back from Online Store
# ============================================================
print("\n" + "="*70)
print("VERIFYING DATA (Reading from Online Store)")
print("="*70)

print("\n5️⃣  Reading user_001 from online store...")

try:
    response = featurestore_runtime.get_record(
        FeatureGroupName=user_feature_group_name,
        RecordIdentifierValueAsString="user_001"
    )
    
    print("   Retrieved record:")
    for feature in response['Record']:
        print(f"      {feature['FeatureName']}: {feature['ValueAsString']}")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n6️⃣  Reading prod_001 from online store...")

try:
    response = featurestore_runtime.get_record(
        FeatureGroupName=product_feature_group_name,
        RecordIdentifierValueAsString="prod_001"
    )
    
    print("   Retrieved record:")
    for feature in response['Record']:
        print(f"      {feature['FeatureName']}: {feature['ValueAsString']}")
        
except Exception as e:
    print(f"   ❌ Error: {e}")

# ============================================================
# Summary
# ============================================================
print("\n" + "="*70)
print("✅ DATA INGESTION COMPLETE!")
print("="*70)
print(f"\nUsers ingested: {user_success}")
print(f"Products ingested: {product_success}")
print("\n📝 Notes:")
print("   - Online store: Available immediately (just verified above)")
print("   - Offline store: Takes ~15 minutes to sync to S3/Athena")
print("\n💡 Next steps:")
print("   1. Query online store for real-time inference")
print("   2. Wait 15 min, then query offline store for training data")
