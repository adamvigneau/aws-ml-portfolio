"""
Query SageMaker Feature Store Online Store for Real-Time Inference
"""
import boto3
import time

# Configuration
region = boto3.Session().region_name
featurestore_runtime = boto3.client('sagemaker-featurestore-runtime', region_name=region)

print(f"Region: {region}")

# Feature group names
user_feature_group_name = "users-feature-group"
product_feature_group_name = "products-feature-group"

print("\n" + "="*70)
print("ONLINE STORE QUERIES (Real-Time Inference)")
print("="*70)

# ============================================================
# Single Record Lookup
# ============================================================
print("\n1️⃣  Single Record Lookup")
print("-" * 40)

def get_user_features(user_id):
    """Get features for a single user"""
    try:
        response = featurestore_runtime.get_record(
            FeatureGroupName=user_feature_group_name,
            RecordIdentifierValueAsString=user_id
        )
        return {f['FeatureName']: f['ValueAsString'] for f in response['Record']}
    except Exception as e:
        return {"error": str(e)}

def get_product_features(product_id):
    """Get features for a single product"""
    try:
        response = featurestore_runtime.get_record(
            FeatureGroupName=product_feature_group_name,
            RecordIdentifierValueAsString=product_id
        )
        return {f['FeatureName']: f['ValueAsString'] for f in response['Record']}
    except Exception as e:
        return {"error": str(e)}

# Test single lookups
print("\nLooking up user_003:")
user_features = get_user_features("user_003")
for key, value in user_features.items():
    if key != "event_time":
        print(f"   {key}: {value}")

print("\nLooking up prod_004:")
product_features = get_product_features("prod_004")
for key, value in product_features.items():
    if key != "event_time":
        print(f"   {key}: {value}")

# ============================================================
# Batch Lookup (Multiple Records)
# ============================================================
print("\n2️⃣  Batch Lookup (Multiple Records)")
print("-" * 40)

def batch_get_features(feature_group_name, record_ids, id_feature_name):
    """Get features for multiple records at once"""
    try:
        identifiers = [
            {
                "FeatureGroupName": feature_group_name,
                "RecordIdentifiersValueAsString": record_ids
            }
        ]
        
        response = featurestore_runtime.batch_get_record(
            Identifiers=identifiers
        )
        
        records = []
        for record in response.get('Records', []):
            record_dict = {f['FeatureName']: f['ValueAsString'] for f in record['Record']}
            records.append(record_dict)
        
        return records
        
    except Exception as e:
        return {"error": str(e)}

# Batch lookup for users
print("\nBatch lookup for users [user_001, user_002, user_005]:")
user_records = batch_get_features(
    user_feature_group_name, 
    ["user_001", "user_002", "user_005"],
    "user_id"
)

for record in user_records:
    print(f"   {record.get('user_id')}: {record.get('membership_tier')} tier, "
          f"{record.get('total_purchases')} purchases, "
          f"${record.get('avg_order_value')} avg order")

# ============================================================
# Simulated Inference Request
# ============================================================
print("\n3️⃣  Simulated Inference Request")
print("-" * 40)
print("\nScenario: User user_002 is viewing product prod_003")
print("Building feature vector for ML model...\n")

# Get user features
user = get_user_features("user_002")
product = get_product_features("prod_003")

# Build feature vector (what you'd send to your ML model)
feature_vector = {
    "user_age": int(user.get("age", 0)),
    "user_membership_tier": user.get("membership_tier", "unknown"),
    "user_total_purchases": int(user.get("total_purchases", 0)),
    "user_avg_order_value": float(user.get("avg_order_value", 0)),
    "product_category": product.get("category", "unknown"),
    "product_price": float(product.get("price", 0)),
    "product_avg_rating": float(product.get("avg_rating", 0)),
    "product_stock_level": int(product.get("stock_level", 0))
}

print("Feature vector for inference:")
for key, value in feature_vector.items():
    print(f"   {key}: {value}")

print("\n   → This feature vector would be sent to your ML model")
print("   → Model predicts: purchase probability, recommended discount, etc.")

# ============================================================
# Latency Test
# ============================================================
print("\n4️⃣  Latency Test (10 lookups)")
print("-" * 40)

latencies = []
for i in range(10):
    start = time.time()
    get_user_features("user_001")
    elapsed = (time.time() - start) * 1000  # Convert to ms
    latencies.append(elapsed)

avg_latency = sum(latencies) / len(latencies)
min_latency = min(latencies)
max_latency = max(latencies)

print(f"\n   Average latency: {avg_latency:.1f} ms")
print(f"   Min latency: {min_latency:.1f} ms")
print(f"   Max latency: {max_latency:.1f} ms")

# ============================================================
# Summary
# ============================================================
print("\n" + "="*70)
print("✅ ONLINE STORE QUERY COMPLETE!")
print("="*70)
print("\n📝 Key Takeaways:")
print("   - Online store provides single-digit millisecond latency")
print("   - Use for real-time inference (recommendations, fraud detection)")
print("   - Supports single and batch lookups")
print("   - Features are always up-to-date (latest ingested values)")
print("\n💡 Production pattern:")
print("   1. User makes request → get user features from online store")
print("   2. Get product/context features from online store")
print("   3. Combine into feature vector")
print("   4. Send to ML model for prediction")
print("   5. Return prediction to user")
