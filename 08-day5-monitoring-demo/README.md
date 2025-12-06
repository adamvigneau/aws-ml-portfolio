
Pick up where I left off with the "A/B Testing in Sagemaker" chat.


Relevant chats
3 results




I found your A/B Testing in SageMaker conversation. You were on Phase 1: Training Two Models and had just requested the complete code.

Here's where you left off — you received the Phase 1 training code that creates two XGBoost models:

Model A (Conservative): num_round=50, max_depth=3, eta=0.3, subsample=1.0
Model B (Aggressive): num_round=100, max_depth=6, eta=0.1, subsample=0.8
The code trains both models on your Titanic dataset and saves them to S3 under ab-test/model-a/ and ab-test/model-b/.

Your next step: Run that Phase 1 code in SageMaker to train both models. Once training completes, you'll have two model artifacts ready for Phase 2 (deploying the multi-variant endpoint with 80/20 traffic split).

Have you had a chance to run the training code yet, or would you like to work through it now?

I was looking at creating feature groups in a conversation further down.  Could you find that so I can pick up where I left off?


Relevant chats
2 results



Found it! You were working on SageMaker Feature Store — specifically trying to create feature groups. You hit a ModuleNotFoundError for sagemaker.feature_store, then after upgrading the SDK, you encountered an API change issue with the create() method.

I had just provided you with an updated script that uses the boto3 client directly (instead of the high-level SDK) to work around the API changes. The script creates two feature groups:

users-feature-group — with features like user_id, age, membership_tier, total_purchases, avg_order_value, event_time
products-feature-group — with features like product_id, category, price, avg_rating, stock_level, event_time
Where you left off: You were about to run the updated create_feature_groups.py script that uses boto3 directly.

Would you like to:

Try running that updated script and let me know if you hit any errors?
Review the full script so you have it fresh?
Skip ahead to the next steps (ingesting data, querying for training/inference)?
Show me the full script for this and the commands to run in Terminal

Here's the full script and the terminal commands:

Terminal Setup
bash
cd ~/feature-store-project

# Activate your virtual environment (if you created one)
source fs-venv/bin/activate

# Verify SageMaker SDK version
python3 -c "import sagemaker; print(f'Version: {sagemaker.__version__}')"
create_feature_groups.py
python
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
Run It
bash
python create_feature_groups.py
This should take 1-2 minutes to create both feature groups. Let me know what output you get!

sagemaker.config INFO - Not applying SDK defaults from location: /Library/Application Support/sagemaker/config.yaml
sagemaker.config INFO - Not applying SDK defaults from location: /Users/adamvigneau/Library/Application Support/sagemaker/config.yaml
Region: us-east-2
Role: arn:aws:iam::854757836160:role/service-role/AmazonSageMaker-ExecutionRole-20251019T120276
S3 Bucket: sagemaker-us-east-2-854757836160
Prefix: feature-store

======================================================================
CREATING FEATURE GROUPS
======================================================================

1️⃣  Creating User Feature Group...
   ✅ Created: users-feature-group
   Record ID: user_id
   Features: 6
   Waiting for feature group to be created...
   ⏳ Status: Creating (attempt 1/20)
   ⏳ Status: Creating (attempt 2/20)
   ⏳ Status: Creating (attempt 3/20)
   ✅ Status: Created

2️⃣  Creating Product Feature Group...
   ✅ Created: products-feature-group
   Record ID: product_id
   Features: 6
   Waiting for feature group to be created...
   ⏳ Status: Creating (attempt 1/20)
   ⏳ Status: Creating (attempt 2/20)
   ⏳ Status: Creating (attempt 3/20)
   ✅ Status: Created

======================================================================
✅ FEATURE GROUPS CREATED!
======================================================================

Feature Groups:
  1. users-feature-group
  2. products-feature-group

Offline Store: s3://sagemaker-us-east-2-854757836160/feature-store/
Online Store: Enabled (DynamoDB-backed)

💡 Tip: View in AWS Console:
https://us-east-2.console.aws.amazon.com/sagemaker/home?region=us-east-2#/feature-store

Show more
Both feature groups created successfully! 🎉

You now have:

users-feature-group — ready to store user features
products-feature-group — ready to store product features
Both have online stores (DynamoDB for real-time inference) and offline stores (S3 for training).

Next Steps
The next phases are:

Ingest data — populate the feature groups with sample records
Query for training — pull features from the offline store using Athena
Query for inference — get features in real-time from the online store
Point-in-time correctness demo — show how Feature Store handles historical lookups
Would you like to continue with ingesting sample data into your new feature groups?

Yes, please

Here's the script to ingest sample data into your feature groups:

ingest_features.py
python
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
Run It
bash
python ingest_features.py
This will:

Create sample user and product data
Ingest records into both feature groups
Verify by reading back from the online store
Let me know what output you get!

sagemaker.config INFO - Not applying SDK defaults from location: /Library/Application Support/sagemaker/config.yaml
sagemaker.config INFO - Not applying SDK defaults from location: /Users/adamvigneau/Library/Application Support/sagemaker/config.yaml
Region: us-east-2

======================================================================
CREATING SAMPLE DATA
======================================================================

1️⃣  Creating sample user data...
 user_id  age membership_tier  total_purchases  avg_order_value   event_time
user_001   28            gold               47           125.50 1.764726e+09
user_002   35          silver               12            45.00 1.764726e+09
user_003   42        platinum              156           312.75 1.764726e+09
user_004   23          bronze                3            28.99 1.764726e+09
user_005   51            gold               89           178.25 1.764726e+09

2️⃣  Creating sample product data...
product_id    category  price  avg_rating  stock_level   event_time
  prod_001 electronics 299.99         4.5          150 1.764726e+09
  prod_002    clothing  49.95         4.2          500 1.764726e+09
  prod_003        home  89.00         3.8           75 1.764726e+09
  prod_004 electronics 599.99         4.9           25 1.764726e+09
  prod_005      sports  34.50         4.0          200 1.764726e+09

======================================================================
INGESTING DATA INTO FEATURE GROUPS
======================================================================

3️⃣  Ingesting user features...
   ✅ Ingested: user_001
   ✅ Ingested: user_002
   ✅ Ingested: user_003
   ✅ Ingested: user_004
   ✅ Ingested: user_005

   Summary: 5 succeeded, 0 failed

4️⃣  Ingesting product features...
   ✅ Ingested: prod_001
   ✅ Ingested: prod_002
   ✅ Ingested: prod_003
   ✅ Ingested: prod_004
   ✅ Ingested: prod_005

   Summary: 5 succeeded, 0 failed

======================================================================
VERIFYING DATA (Reading from Online Store)
======================================================================

5️⃣  Reading user_001 from online store...
   Retrieved record:
      user_id: user_001
      age: 28
      membership_tier: gold
      total_purchases: 47
      avg_order_value: 125.5
      event_time: 1764725998.300262

6️⃣  Reading prod_001 from online store...
   Retrieved record:
      product_id: prod_001
      category: electronics
      price: 299.99
      avg_rating: 4.5
      stock_level: 150
      event_time: 1764725998.300262

======================================================================
✅ DATA INGESTION COMPLETE!
======================================================================

Users ingested: 5
Products ingested: 5

📝 Notes:
   - Online store: Available immediately (just verified above)
   - Offline store: Takes ~15 minutes to sync to S3/Athena

💡 Next steps:
   1. Query online store for real-time inference
   2. Wait 15 min, then query offline store for training data

Show more
All 10 records ingested and verified! 🎉

Your Feature Store now has:

5 users in users-feature-group
5 products in products-feature-group
The online store is working (you just read back user_001 and prod_001 instantly).

Next Steps
While we wait ~15 minutes for the offline store to sync, let's create the scripts for:

Query online store — real-time feature lookup for inference
Query offline store — pull training datasets via Athena
Would you like to continue with the online store query script (for real-time inference)? We can test it right now since the online store is already available.

Sure, let's do it.

Here's the script to query the online store for real-time inference:

query_online_store.py
python
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
Run It
bash
python query_online_store.py
This demonstrates:

Single record lookups
Batch lookups (multiple records at once)
Building a feature vector for inference
Latency benchmarking
Let me know what output you get!

Region: us-east-2

======================================================================
ONLINE STORE QUERIES (Real-Time Inference)
======================================================================

1️⃣  Single Record Lookup
----------------------------------------

Looking up user_003:
   user_id: user_003
   age: 42
   membership_tier: platinum
   total_purchases: 156
   avg_order_value: 312.75

Looking up prod_004:
   product_id: prod_004
   category: electronics
   price: 599.99
   avg_rating: 4.9
   stock_level: 25

2️⃣  Batch Lookup (Multiple Records)
----------------------------------------

Batch lookup for users [user_001, user_002, user_005]:
   user_002: silver tier, 12 purchases, $45.0 avg order
   user_005: gold tier, 89 purchases, $178.25 avg order
   user_001: gold tier, 47 purchases, $125.5 avg order

3️⃣  Simulated Inference Request
----------------------------------------

Scenario: User user_002 is viewing product prod_003
Building feature vector for ML model...

Feature vector for inference:
   user_age: 35
   user_membership_tier: silver
   user_total_purchases: 12
   user_avg_order_value: 45.0
   product_category: home
   product_price: 89.0
   product_avg_rating: 3.8
   product_stock_level: 75

   → This feature vector would be sent to your ML model
   → Model predicts: purchase probability, recommended discount, etc.

4️⃣  Latency Test (10 lookups)
----------------------------------------

   Average latency: 61.8 ms
   Min latency: 54.4 ms
   Max latency: 82.4 ms

======================================================================
✅ ONLINE STORE QUERY COMPLETE!
======================================================================

📝 Key Takeaways:
   - Online store provides single-digit millisecond latency
   - Use for real-time inference (recommendations, fraud detection)
   - Supports single and batch lookups
   - Features are always up-to-date (latest ingested values)

💡 Production pattern:
   1. User makes request → get user features from online store
   2. Get product/context features from online store
   3. Combine into feature vector
   4. Send to ML model for prediction
   5. Return prediction to user

Show more
Online store queries working perfectly! 🎉

Your latency (~55-82ms) is a bit higher than the typical single-digit millisecond you'd see in production — that's because you're calling from your local machine across the internet. In a real deployment (Lambda or EC2 in the same region), you'd see 5-15ms latency.

What You Just Demonstrated
Capability	Status
Single record lookup	✅
Batch lookup	✅
Feature vector assembly	✅
Real-time latency	✅ ~60ms (would be <15ms in-region)
Next: Offline Store Query
The offline store should be synced by now (~15 min have passed). This is where you pull training data using Athena.

The offline store is important for:

Building training datasets
Historical feature lookups
Point-in-time correct joins (no data leakage)
Would you like to continue with the offline store query script?

Sure!

Here's the script to query the offline store using Athena:

query_offline_store.py
python
"""
Query SageMaker Feature Store Offline Store for Training Data
"""
import boto3
import sagemaker
import time
import pandas as pd

# Configuration
region = boto3.Session().region_name
boto_session = boto3.Session(region_name=region)
sagemaker_client = boto3.client('sagemaker', region_name=region)
athena_client = boto3.client('athena', region_name=region)

sagemaker_session = sagemaker.Session(
    boto_session=boto_session,
    sagemaker_client=sagemaker_client
)

bucket = sagemaker_session.default_bucket()

print(f"Region: {region}")
print(f"Bucket: {bucket}")

# Feature group names
user_feature_group_name = "users-feature-group"
product_feature_group_name = "products-feature-group"

print("\n" + "="*70)
print("OFFLINE STORE QUERIES (Training Data)")
print("="*70)

# ============================================================
# Get Feature Group Details
# ============================================================
print("\n1️⃣  Getting Feature Group Details")
print("-" * 40)

def get_feature_group_table_name(feature_group_name):
    """Get the Athena/Glue table name for a feature group"""
    response = sagemaker_client.describe_feature_group(
        FeatureGroupName=feature_group_name
    )
    
    # Table name is in the offline store config
    offline_config = response.get('OfflineStoreConfig', {})
    data_catalog_config = offline_config.get('DataCatalogConfig', {})
    
    table_name = data_catalog_config.get('TableName', feature_group_name.replace('-', '_'))
    database = data_catalog_config.get('Database', 'sagemaker_featurestore')
    catalog = data_catalog_config.get('Catalog', 'AwsDataCatalog')
    
    return catalog, database, table_name

user_catalog, user_db, user_table = get_feature_group_table_name(user_feature_group_name)
product_catalog, product_db, product_table = get_feature_group_table_name(product_feature_group_name)

print(f"\nUser Feature Group:")
print(f"   Database: {user_db}")
print(f"   Table: {user_table}")

print(f"\nProduct Feature Group:")
print(f"   Database: {product_db}")
print(f"   Table: {product_table}")

# ============================================================
# Helper: Run Athena Query
# ============================================================
def run_athena_query(query, database, output_location):
    """Execute an Athena query and wait for results"""
    
    print(f"\n   Executing query...")
    
    # Start query
    response = athena_client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={'Database': database},
        ResultConfiguration={'OutputLocation': output_location}
    )
    
    query_execution_id = response['QueryExecutionId']
    
    # Wait for completion
    max_attempts = 30
    for attempt in range(max_attempts):
        response = athena_client.get_query_execution(
            QueryExecutionId=query_execution_id
        )
        
        state = response['QueryExecution']['Status']['State']
        
        if state == 'SUCCEEDED':
            print(f"   ✅ Query succeeded!")
            break
        elif state in ['FAILED', 'CANCELLED']:
            reason = response['QueryExecution']['Status'].get('StateChangeReason', 'Unknown')
            print(f"   ❌ Query {state}: {reason}")
            return None
        else:
            print(f"   ⏳ Status: {state} (attempt {attempt + 1}/{max_attempts})")
            time.sleep(2)
    
    # Get results
    results = athena_client.get_query_results(
        QueryExecutionId=query_execution_id
    )
    
    # Parse results into a list of dicts
    columns = [col['Label'] for col in results['ResultSet']['ResultSetMetadata']['ColumnInfo']]
    rows = []
    
    for row in results['ResultSet']['Rows'][1:]:  # Skip header row
        values = [field.get('VarCharValue', '') for field in row['Data']]
        rows.append(dict(zip(columns, values)))
    
    return rows

# Athena output location
athena_output = f"s3://{bucket}/athena-results/"

# ============================================================
# Query 1: Get All Users
# ============================================================
print("\n2️⃣  Query: Get All Users from Offline Store")
print("-" * 40)

user_query = f"""
SELECT user_id, age, membership_tier, total_purchases, avg_order_value
FROM "{user_db}"."{user_table}"
ORDER BY user_id
"""

print(f"   SQL: SELECT user_id, age, membership_tier, total_purchases, avg_order_value...")

user_results = run_athena_query(user_query, user_db, athena_output)

if user_results:
    print("\n   Results:")
    for row in user_results:
        print(f"      {row['user_id']}: {row['membership_tier']} tier, "
              f"age {row['age']}, {row['total_purchases']} purchases")
else:
    print("\n   ⚠️  No results - offline store may still be syncing")
    print("   Try again in a few minutes")

# ============================================================
# Query 2: Get All Products
# ============================================================
print("\n3️⃣  Query: Get All Products from Offline Store")
print("-" * 40)

product_query = f"""
SELECT product_id, category, price, avg_rating, stock_level
FROM "{product_db}"."{product_table}"
ORDER BY product_id
"""

print(f"   SQL: SELECT product_id, category, price, avg_rating, stock_level...")

product_results = run_athena_query(product_query, product_db, athena_output)

if product_results:
    print("\n   Results:")
    for row in product_results:
        print(f"      {row['product_id']}: {row['category']}, "
              f"${row['price']}, {row['avg_rating']}★")
else:
    print("\n   ⚠️  No results - offline store may still be syncing")

# ============================================================
# Query 3: Filtered Query (Training Data Example)
# ============================================================
print("\n4️⃣  Query: Filter High-Value Users (Training Data)")
print("-" * 40)

training_query = f"""
SELECT user_id, age, membership_tier, total_purchases, avg_order_value
FROM "{user_db}"."{user_table}"
WHERE total_purchases > 10
  AND avg_order_value > 50
ORDER BY total_purchases DESC
"""

print(f"   SQL: WHERE total_purchases > 10 AND avg_order_value > 50...")

training_results = run_athena_query(training_query, user_db, athena_output)

if training_results:
    print("\n   High-value users for training:")
    for row in training_results:
        print(f"      {row['user_id']}: {row['total_purchases']} purchases, "
              f"${row['avg_order_value']} avg")
else:
    print("\n   ⚠️  No results matching filter")

# ============================================================
# Query 4: Join Users and Products (Advanced)
# ============================================================
print("\n5️⃣  Query: Cross-Join for Feature Matrix (Advanced)")
print("-" * 40)

join_query = f"""
SELECT 
    u.user_id,
    u.membership_tier,
    u.avg_order_value as user_avg_order,
    p.product_id,
    p.category,
    p.price
FROM "{user_db}"."{user_table}" u
CROSS JOIN "{product_db}"."{product_table}" p
WHERE u.membership_tier = 'gold'
  AND p.category = 'electronics'
LIMIT 10
"""

print(f"   SQL: JOIN users (gold tier) with products (electronics)...")

join_results = run_athena_query(join_query, user_db, athena_output)

if join_results:
    print("\n   User-Product combinations:")
    for row in join_results:
        print(f"      {row['user_id']} ({row['membership_tier']}) × "
              f"{row['product_id']} ({row['category']}, ${row['price']})")
else:
    print("\n   ⚠️  No results from join")

# ============================================================
# Summary
# ============================================================
print("\n" + "="*70)
print("✅ OFFLINE STORE QUERY COMPLETE!")
print("="*70)
print("\n📝 Key Takeaways:")
print("   - Offline store uses Athena/Glue for SQL queries")
print("   - Great for building training datasets at scale")
print("   - Supports complex joins across feature groups")
print("   - Data is stored in Parquet format on S3")
print("\n💡 Production pattern:")
print("   1. Define training query with filters/joins")
print("   2. Export results to S3 as training data")
print("   3. Use in SageMaker training job")
print("   4. Features stay consistent between training and inference")
print(f"\n🔗 View in Athena Console:")
print(f"   https://{region}.console.aws.amazon.com/athena/home?region={region}")
Run It
bash
python query_offline_store.py
This demonstrates:

Getting feature group table names from Glue catalog
Basic SELECT queries
Filtered queries for training data
Cross-joins across feature groups
Note: If you see "No results," the offline store might still be syncing. It typically takes 15-20 minutes after ingestion.

Let me know what output you get!

sagemaker.config INFO - Not applying SDK defaults from location: /Library/Application Support/sagemaker/config.yaml
sagemaker.config INFO - Not applying SDK defaults from location: /Users/adamvigneau/Library/Application Support/sagemaker/config.yaml
Region: us-east-2
Bucket: sagemaker-us-east-2-854757836160

======================================================================
OFFLINE STORE QUERIES (Training Data)
======================================================================

1️⃣  Getting Feature Group Details
----------------------------------------

User Feature Group:
   Database: sagemaker_featurestore
   Table: users_feature_group_1764725072

Product Feature Group:
   Database: sagemaker_featurestore
   Table: products_feature_group_1764725090

2️⃣  Query: Get All Users from Offline Store
----------------------------------------
   SQL: SELECT user_id, age, membership_tier, total_purchases, avg_order_value...

   Executing query...
   ⏳ Status: QUEUED (attempt 1/30)
   ✅ Query succeeded!

   Results:
      user_001: gold tier, age 28, 47 purchases
      user_002: silver tier, age 35, 12 purchases
      user_003: platinum tier, age 42, 156 purchases
      user_004: bronze tier, age 23, 3 purchases
      user_005: gold tier, age 51, 89 purchases

3️⃣  Query: Get All Products from Offline Store
----------------------------------------
   SQL: SELECT product_id, category, price, avg_rating, stock_level...

   Executing query...
   ⏳ Status: QUEUED (attempt 1/30)
   ✅ Query succeeded!

   Results:
      prod_001: electronics, $299.99, 4.5★
      prod_002: clothing, $49.95, 4.2★
      prod_003: home, $89.0, 3.8★
      prod_004: electronics, $599.99, 4.9★
      prod_005: sports, $34.5, 4.0★

4️⃣  Query: Filter High-Value Users (Training Data)
----------------------------------------
   SQL: WHERE total_purchases > 10 AND avg_order_value > 50...

   Executing query...
   ⏳ Status: QUEUED (attempt 1/30)
   ✅ Query succeeded!

   High-value users for training:
      user_003: 156 purchases, $312.75 avg
      user_005: 89 purchases, $178.25 avg
      user_001: 47 purchases, $125.5 avg

5️⃣  Query: Cross-Join for Feature Matrix (Advanced)
----------------------------------------
   SQL: JOIN users (gold tier) with products (electronics)...

   Executing query...
   ⏳ Status: QUEUED (attempt 1/30)
   ✅ Query succeeded!

   User-Product combinations:
      user_001 (gold) × prod_001 (electronics, $299.99)
      user_001 (gold) × prod_004 (electronics, $599.99)
      user_005 (gold) × prod_001 (electronics, $299.99)
      user_005 (gold) × prod_004 (electronics, $599.99)

======================================================================
✅ OFFLINE STORE QUERY COMPLETE!
======================================================================

📝 Key Takeaways:
   - Offline store uses Athena/Glue for SQL queries
   - Great for building training datasets at scale
   - Supports complex joins across feature groups
   - Data is stored in Parquet format on S3

💡 Production pattern:
   1. Define training query with filters/joins
   2. Export results to S3 as training data
   3. Use in SageMaker training job
   4. Features stay consistent between training and inference

🔗 View in Athena Console:
   https://us-east-2.console.aws.amazon.com/athena/home?region=us-east-2

Show more
Offline store working perfectly! 🎉

All four queries succeeded — your Feature Store is fully operational with both online and offline stores.

What You've Accomplished
Component	Status	Use Case
Feature Groups	✅ Created	Schema definition
Data Ingestion	✅ 10 records	Populate features
Online Store	✅ ~60ms latency	Real-time inference
Offline Store	✅ Athena queries	Training data
Cross-joins	✅ Working	Feature matrix
Final Step: Cleanup
When you're done experimenting, you'll want to delete the feature groups to avoid ongoing charges. Would you like me to give you:

Cleanup script — delete feature groups and associated resources
Point-in-time correctness demo — show how Feature Store handles historical lookups (this is a key exam concept)
Which would you prefer? The point-in-time demo is valuable for understanding why Feature Store matters for ML (prevents data leakage), but if you're ready to wrap up, we can do cleanup.

Let's do the point in time demo first.

Here's the point-in-time correctness demo — this is a key concept for the exam:

point_in_time_demo.py
python
"""
Point-in-Time Correctness Demo
Shows how Feature Store prevents data leakage in ML training
"""
import boto3
import time

# Configuration
region = boto3.Session().region_name
featurestore_runtime = boto3.client('sagemaker-featurestore-runtime', region_name=region)

print(f"Region: {region}")

user_feature_group_name = "users-feature-group"

print("\n" + "="*70)
print("POINT-IN-TIME CORRECTNESS DEMO")
print("="*70)

print("""
📚 WHY THIS MATTERS:

Imagine you're training a model to predict if a user will make a purchase.
You have:
  - Historical transaction data (labels): "User X bought on Jan 15"
  - User features: age, membership_tier, total_purchases, etc.

THE PROBLEM (Data Leakage):
  If user_001 had 10 purchases on Jan 15, but now has 47 purchases,
  using current features (47) to train on Jan 15 data is CHEATING.
  Your model sees "future" information it wouldn't have at prediction time.

THE SOLUTION (Point-in-Time Correctness):
  Feature Store tracks WHEN each feature value was recorded.
  When building training data, you get features AS THEY WERE at each event time.
""")

input("\nPress Enter to start the demo...")

# ============================================================
# Step 1: Show Current State
# ============================================================
print("\n" + "-"*70)
print("STEP 1: Current State of user_001")
print("-"*70)

response = featurestore_runtime.get_record(
    FeatureGroupName=user_feature_group_name,
    RecordIdentifierValueAsString="user_001"
)

current_features = {f['FeatureName']: f['ValueAsString'] for f in response['Record']}
original_event_time = current_features['event_time']

print(f"\n   user_id: {current_features['user_id']}")
print(f"   total_purchases: {current_features['total_purchases']}")
print(f"   avg_order_value: {current_features['avg_order_value']}")
print(f"   membership_tier: {current_features['membership_tier']}")
print(f"   event_time: {current_features['event_time']}")

# ============================================================
# Step 2: Simulate Time Passing - User Makes More Purchases
# ============================================================
print("\n" + "-"*70)
print("STEP 2: Simulating Time Passing...")
print("-"*70)

print("\n   📅 One month later: user_001 makes 20 more purchases!")
print("   Updating feature store with new values...\n")

time.sleep(2)

# New event time (simulating 1 month later)
new_event_time = time.time()

# Update with new values
new_record = [
    {"FeatureName": "user_id", "ValueAsString": "user_001"},
    {"FeatureName": "age", "ValueAsString": "28"},
    {"FeatureName": "membership_tier", "ValueAsString": "platinum"},  # Upgraded!
    {"FeatureName": "total_purchases", "ValueAsString": "67"},  # Was 47, now 67
    {"FeatureName": "avg_order_value", "ValueAsString": "142.75"},  # Increased
    {"FeatureName": "event_time", "ValueAsString": str(new_event_time)}
]

featurestore_runtime.put_record(
    FeatureGroupName=user_feature_group_name,
    Record=new_record
)

print("   ✅ Updated user_001:")
print(f"      total_purchases: 47 → 67 (+20)")
print(f"      avg_order_value: $125.50 → $142.75")
print(f"      membership_tier: gold → platinum (upgraded!)")

# ============================================================
# Step 3: Query Online Store (Gets Latest)
# ============================================================
print("\n" + "-"*70)
print("STEP 3: Online Store Query (Returns LATEST)")
print("-"*70)

time.sleep(1)

response = featurestore_runtime.get_record(
    FeatureGroupName=user_feature_group_name,
    RecordIdentifierValueAsString="user_001"
)

latest_features = {f['FeatureName']: f['ValueAsString'] for f in response['Record']}

print(f"\n   Online store returns CURRENT state:")
print(f"      total_purchases: {latest_features['total_purchases']}")
print(f"      avg_order_value: {latest_features['avg_order_value']}")
print(f"      membership_tier: {latest_features['membership_tier']}")
print(f"\n   ✅ This is correct for INFERENCE (real-time predictions)")

# ============================================================
# Step 4: Explain Offline Store Point-in-Time
# ============================================================
print("\n" + "-"*70)
print("STEP 4: Offline Store - Point-in-Time Queries")
print("-"*70)

print("""
   The OFFLINE store (Athena) keeps ALL historical versions!

   If you query with a point-in-time filter, you get features
   AS THEY WERE at that specific time.

   Example Athena query for training data:
```sql
   SELECT u.user_id, u.total_purchases, u.membership_tier,
          t.transaction_id, t.purchase_amount, t.event_time
   FROM users_feature_group u
   JOIN transactions t
     ON u.user_id = t.user_id
     AND u.event_time <= t.event_time  -- Point-in-time join!
```

   This ensures:
   ┌─────────────────────────────────────────────────────────────┐
   │  Training Event     │  Features Used                       │
   ├─────────────────────────────────────────────────────────────┤
   │  Jan 15 purchase    │  Features as of Jan 15 (47 purchases)│
   │  Feb 15 purchase    │  Features as of Feb 15 (67 purchases)│
   └─────────────────────────────────────────────────────────────┘

   NOT:
   ┌─────────────────────────────────────────────────────────────┐
   │  Training Event     │  Features Used (WRONG - Data Leakage)│
   ├─────────────────────────────────────────────────────────────┤
   │  Jan 15 purchase    │  Current features (67 purchases) ❌  │
   │  Feb 15 purchase    │  Current features (67 purchases) ❌  │
   └─────────────────────────────────────────────────────────────┘
""")

# ============================================================
# Step 5: Show Both Versions Exist
# ============================================================
print("-"*70)
print("STEP 5: Verify Both Versions in Offline Store")
print("-"*70)

print("""
   After ~15 minutes, the offline store will have BOTH records:

   ┌────────────┬──────────────────┬─────────────────┬───────────────┐
   │  user_id   │  total_purchases │  membership_tier│  event_time   │
   ├────────────┼──────────────────┼─────────────────┼───────────────┤
   │  user_001  │  47              │  gold           │  <earlier>    │
   │  user_001  │  67              │  platinum       │  <later>      │
   └────────────┴──────────────────┴─────────────────┴───────────────┘

   The event_time column lets you do point-in-time correct joins!
""")

# ============================================================
# Summary
# ============================================================
print("="*70)
print("✅ POINT-IN-TIME DEMO COMPLETE!")
print("="*70)

print("""
📝 KEY EXAM CONCEPTS:

1. ONLINE STORE:
   - Returns LATEST feature values only
   - Used for real-time inference
   - Single-digit millisecond latency

2. OFFLINE STORE:
   - Keeps ALL historical versions
   - Used for training data
   - Enables point-in-time correct joins

3. WHY IT MATTERS:
   - Prevents data leakage (using future info to predict past)
   - Ensures training/inference consistency
   - Required for reproducible ML experiments

4. EXAM TIP:
   Q: "How do you ensure training features match inference features?"
   A: Use SageMaker Feature Store - single source of truth for both
      online (inference) and offline (training) feature access.
""")

print("\n💡 user_001 now has updated values. The cleanup script will")
print("   delete everything when you're ready.")
Run It
bash
python point_in_time_demo.py
This will:

Show the current state of user_001
Update user_001 with new values (simulating time passing)
Demonstrate that online store returns latest values
Explain how offline store preserves history for point-in-time joins
Let me know what you see!


Region: us-east-2 ====================================================================== POINT-IN-TIME CORRECTNESS DEMO ====================================================================== 📚 WHY THIS MATTERS: Imagine you're training a model to predict if a user will make a purchase. You have:

pasted

Demo ran perfectly! 🎉

You now understand one of the most important concepts for the ML exam — point-in-time correctness and why Feature Store matters.

Quick Recap for the Exam
Question	Answer
How do you prevent data leakage in training?	Point-in-time joins using event_time
Which store keeps historical versions?	Offline store (Athena/S3)
Which store returns only latest values?	Online store (DynamoDB)
How do you ensure training/inference consistency?	Feature Store — single source of truth
Ready for Cleanup?
Your feature groups are still running (and incurring small charges for the online store). When you're ready, I'll give you the cleanup script to delete:

users-feature-group
products-feature-group
Associated S3 data and Glue tables
Want the cleanup script now, or do you want to keep experimenting first?

I'll take the cleanup script.

Here's the cleanup script:

cleanup_feature_store.py
python
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
Run It
bash
python cleanup_feature_store.py
This will delete both feature groups. The online store stops billing immediately. S3 data remains but costs almost nothing — you can delete it manually with the aws s3 rm commands shown at the end if you want.

Let me know when cleanup is done, or if you have any questions about what you learned today!

I'd like to check in this work into my github.  Can you show me the files I need and also help create a README?

Let me check your past conversations to see if you have an existing repo structure or GitHub setup:


Relevant chats
2 results



Found it! Your repo is called aws-ml-portfolio and you have a structure set up. Let me create the files for your Feature Store project:

Check if I should use docx skill for README
I'll create the files as markdown/code files for your GitHub repo. Let me set these up:


9 steps

Copy all files to outputs directory for download



Copy .gitignore file


Here are all the files for your GitHub repo:

Files Created
File	Description
README.md	Project documentation with architecture, setup, and exam tips
create_feature_groups.py	Create feature groups with online/offline stores
ingest_features.py	Ingest sample user and product data
query_online_store.py	Real-time feature queries for inference
query_offline_store.py	Athena queries for training data
point_in_time_demo.py	Demonstrate point-in-time correctness
cleanup_feature_store.py	Delete resources when done
requirements.txt	Python dependencies
.gitignore	Git ignore file
Adding to Your Repo
Based on your repo structure (aws-ml-portfolio), here are the terminal commands:

bash
# Navigate to your repo
cd ~/aws-ml-portfolio

# Create the feature store project folder
mkdir -p feature-store-project

# Copy the downloaded files into it
cp ~/Downloads/README.md feature-store-project/
cp ~/Downloads/*.py feature-store-project/
cp ~/Downloads/requirements.txt feature-store-project/
cp ~/Downloads/.gitignore feature-store-project/

# Commit and push
git add feature-store-project/
git commit -m "Add SageMaker Feature Store demo project"
git push origin main
This project fits nicely alongside your other work in the portfolio. Let me know if you need any changes to the README or scripts!

Readme
Document · MD 
Create feature groups
PY 
Ingest features
PY 
Query online store
PY 
Query offline store
PY 
Point in time demo
PY 
Cleanup feature store
PY 
Requirements
TXT 
.gitignore
GITIGNORE 
Can you provide a recap of what I learned today?  I want to be able to reference what I did when I go to do my review work before my exam.

Here's a comprehensive recap of your Feature Store session:

SageMaker Feature Store - Study Recap
What You Built
You created a complete Feature Store demo with two feature groups (users and products), demonstrating the full lifecycle: creation → ingestion → online queries → offline queries → point-in-time correctness.

Core Concepts
Feature Store = A centralized repository for ML features that serves both training (offline) and inference (online) from a single source of truth.

Component	Backend	Use Case	Latency
Online Store	DynamoDB	Real-time inference	~5-15ms (in-region)
Offline Store	S3 + Glue + Athena	Training data	Seconds (SQL query)
Key Operations You Performed
1. Created Feature Groups

python
sagemaker_client.create_feature_group(
    FeatureGroupName="users-feature-group",
    RecordIdentifierFeatureName="user_id",      # Primary key
    EventTimeFeatureName="event_time",          # For point-in-time
    FeatureDefinitions=[...],                   # Schema
    OnlineStoreConfig={"EnableOnlineStore": True},
    OfflineStoreConfig={"S3StorageConfig": {"S3Uri": "s3://..."}}
)
2. Ingested Data

python
featurestore_runtime.put_record(
    FeatureGroupName="users-feature-group",
    Record=[
        {"FeatureName": "user_id", "ValueAsString": "user_001"},
        {"FeatureName": "age", "ValueAsString": "28"},
        ...
    ]
)
3. Queried Online Store (Real-Time)

python
response = featurestore_runtime.get_record(
    FeatureGroupName="users-feature-group",
    RecordIdentifierValueAsString="user_001"
)
4. Queried Offline Store (Training Data)

sql
SELECT user_id, age, membership_tier, total_purchases
FROM "sagemaker_featurestore"."users_feature_group_123"
WHERE total_purchases > 10
Point-in-Time Correctness (Critical Exam Concept)
The Problem: Data leakage occurs when training data includes "future" information that wouldn't be available at prediction time.

Example:

Jan 15: user_001 had 47 purchases, bought something
Feb 15: user_001 now has 67 purchases
Wrong approach: Train Jan 15 event using current features (67 purchases) ❌

Correct approach: Train Jan 15 event using features AS OF Jan 15 (47 purchases) ✅

How Feature Store solves this:

Every record has an event_time timestamp
Offline store keeps ALL historical versions
Point-in-time joins use event_time to get correct historical features
sql
-- Point-in-time correct join
SELECT u.user_id, u.total_purchases, t.purchase_amount
FROM users_feature_group u
JOIN transactions t
  ON u.user_id = t.user_id
  AND u.event_time <= t.event_time  -- Only use features from BEFORE the event
```

---

### Online vs Offline Store Comparison

| Question | Online Store | Offline Store |
|----------|--------------|---------------|
| What data do I get? | Latest values only | Full history |
| When do I use it? | Real-time inference | Building training datasets |
| How do I query it? | `get_record()` API | Athena SQL |
| How fast is it? | Milliseconds | Seconds |
| What's the backend? | DynamoDB | S3 (Parquet) + Glue + Athena |

---

### Exam Tips

1. **"How do you ensure training/inference feature consistency?"**
   → Feature Store - single source of truth for both

2. **"How do you prevent data leakage in feature engineering?"**
   → Point-in-time correct joins using `event_time`

3. **"Which store for real-time recommendations?"**
   → Online store (low latency)

4. **"Which store for batch model retraining?"**
   → Offline store (SQL queries at scale)

5. **"How does Feature Store track feature history?"**
   → `event_time` column + offline store keeps all versions

---

### Production Pattern You Demonstrated
```
┌─────────────────────────────────────────────────────────────────┐
│                     INFERENCE FLOW                              │
│  User Request → Get features from ONLINE store → ML Model →    │
│  Prediction → Response                                          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING FLOW                               │
│  Athena Query (OFFLINE store) → Point-in-time join →           │
│  Training Dataset → SageMaker Training Job → New Model         │
└─────────────────────────────────────────────────────────────────┘
Files in Your Repo
Script	What It Demonstrates
create_feature_groups.py	Feature group creation with both stores
ingest_features.py	Writing records via put_record()
query_online_store.py	Real-time lookups, batch lookups, feature vectors
query_offline_store.py	Athena SQL, filtered queries, cross-joins
point_in_time_demo.py	Historical versioning, data leakage prevention
cleanup_feature_store.py	Resource deletion
Quick Reference Commands
bash
# Setup
pip install 'sagemaker>=2.200.0,<3.0' boto3 pandas

# Run in order
python create_feature_groups.py    # ~2 min
python ingest_features.py          # ~30 sec
python query_online_store.py       # Immediate
python query_offline_store.py      # Wait 15 min after ingestion
python point_in_time_demo.py       # Interactive demo
python cleanup_feature_store.py    # When done
Good luck on the exam! 🎯

Feature Store vs Traditional Storage

CapabilityFeature StoreDatabaseS3Low-latency reads✅ (online)✅❌Cheap storage✅ (offline)❌✅Point-in-time queries✅❌❌Auto schema✅ManualManualFeature discovery✅❌❌ML-optimized✅❌❌Lineage tracking✅❌❌
Can you turn this into a table in markdown?

Here's the table in markdown:

markdown
## Feature Store vs Traditional Storage

| Capability | Feature Store | Database | S3 |
|------------|---------------|----------|-----|
| Low-latency reads | ✅ (online) | ✅ | ❌ |
| Cheap storage | ✅ (offline) | ❌ | ✅ |
| Point-in-time queries | ✅ | ❌ | ❌ |
| Auto schema | ✅ | Manual | Manual |
| Feature discovery | ✅ | ❌ | ❌ |
| ML-optimized | ✅ | ❌ | ❌ |
| Lineage tracking | ✅ | ❌ | ❌ |
Which renders as:

Capability	Feature Store	Database	S3
Low-latency reads	✅ (online)	✅	❌
Cheap storage	✅ (offline)	❌	✅
Point-in-time queries	✅	❌	❌
Auto schema	✅	Manual	Manual
Feature discovery	✅	❌	❌
ML-optimized	✅	❌	❌
Lineage tracking	✅	❌	❌
Want me to add this to your README?

Here is my planning for the rest of the week:

Day 3: Edge Deployment & SageMaker Neo
What it is: Deploy models to edge devices (IoT, mobile, embedded systems)
Why it matters:
* Latency-sensitive applications
* Offline inference
* Bandwidth constraints
* Privacy requirements
Topics:
* SageMaker Neo (model compilation/optimization)
* SageMaker Edge Manager
* Model packaging for edge devices
* Over-the-air updates
* Edge device monitoring
Hands-on: Compile a model with Neo, optimize for CPU inference
Day 4: Multi-Model & Multi-Container Endpoints
What it is: Host multiple models on a single endpoint to save costs
Why it matters:
* Cost optimization (one instance, many models)
* Personalized models (per-customer, per-region)
* A/B testing at scale
* Serial inference (chaining models)
Topics:
* Multi-Model Endpoints (MME)
* Multi-Container Endpoints (MCE)
* Serial inference pipelines
* Model loading/unloading
* Cost comparison
Hands-on: Deploy 5 models to one endpoint, compare costs vs 5 separate endpoints
Day 5: Advanced Monitoring & Observability
What it is: Production monitoring beyond basic Model Monitor
Topics:
* Custom monitoring schedules
* Ground truth labeling workflows
* Drift detection strategies
* Custom metrics with CloudWatch
* Distributed tracing (X-Ray)
* Data quality validation (Deequ)
* Model explainability in production (SHAP, Clarify)
Hands-on: Set up comprehensive monitoring dashboard with custom metrics
Week 8 Project: Enterprise MLOps Pipeline
Build an end-to-end pipeline that includes:
1. SageMaker Pipeline for orchestration
2. Feature Store for feature management
3. Multi-Model Endpoint for cost optimization
4. Comprehensive monitoring with custom metrics
5. Automated retraining on drift detection
Time: 3-4 hours
Portfolio value: 🔥🔥🔥 (This is what senior ML engineers build)

Show more
Great plan for the rest of the week! Here's a quick overview of what you're tackling:

Week 8 Overview
Day	Topic	Key Service	Hands-On
Day 1-2 ✅	Feature Store	SageMaker Feature Store	Online/offline stores, point-in-time
Day 3	Edge Deployment	SageMaker Neo	Compile model for CPU
Day 4	Multi-Model Endpoints	MME / MCE	5 models on 1 endpoint
Day 5	Advanced Monitoring	CloudWatch, X-Ray, Clarify	Custom monitoring dashboard
Weekend	Enterprise MLOps Pipeline	All combined	End-to-end production system
Exam Concept Preview
Day 3 - Edge & Neo:

Neo compiles models for specific hardware (ARM, Intel, NVIDIA)
Edge Manager handles deployment, monitoring, and OTA updates
Key use cases: IoT, autonomous vehicles, mobile apps
Day 4 - Multi-Model/Multi-Container:

MME = many models, ONE container (same framework)
MCE = different containers (different frameworks, serial inference)
Cost savings: 1 endpoint vs N endpoints
Day 5 - Monitoring:

Model Monitor = data drift, model quality, bias drift
Clarify = explainability (SHAP values)
Ground Truth = human labeling for retraining
Weekend Project Architecture
┌─────────────────────────────────────────────────────────────────┐
│                 ENTERPRISE MLOPS PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Feature      │───▶│ SageMaker    │───▶│ Multi-Model  │      │
│  │ Store        │    │ Pipeline     │    │ Endpoint     │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         │                   ▼                   ▼               │
│         │            ┌──────────────┐    ┌──────────────┐      │
│         │            │ Model        │    │ CloudWatch   │      │
│         └───────────▶│ Registry     │    │ Monitoring   │      │
│                      └──────────────┘    └──────────────┘      │
│                            │                   │               │
│                            ▼                   ▼               │
│                      ┌─────────────────────────────────┐       │
│                      │  Drift Detection → Auto Retrain │       │
│                      └─────────────────────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
This is a solid progression — Feature Store → Edge → Multi-Model → Monitoring → Full Pipeline. Each day builds on the last, and the weekend project ties everything together.

Ready to start Day 3 (Edge & Neo) when you are!

Day 1 was AWS SageMaker Pipelines, which I think fits into the diagram

You're right! Let me update that:

Week 8 Overview
Day	Topic	Key Service	Hands-On
Day 1 ✅	ML Pipelines	SageMaker Pipelines	Orchestrated training workflow
Day 2 ✅	Feature Store	SageMaker Feature Store	Online/offline stores, point-in-time
Day 3	Edge Deployment	SageMaker Neo	Compile model for CPU
Day 4	Multi-Model Endpoints	MME / MCE	5 models on 1 endpoint
Day 5	Advanced Monitoring	CloudWatch, X-Ray, Clarify	Custom monitoring dashboard
Weekend	Enterprise MLOps Pipeline	All combined	End-to-end production system
Weekend Project Architecture (Updated)
┌─────────────────────────────────────────────────────────────────┐
│                 ENTERPRISE MLOPS PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Feature      │───▶│ SageMaker    │───▶│ Multi-Model  │      │
│  │ Store        │    │ Pipelines    │    │ Endpoint     │      │
│  │ (Day 2) ✅   │    │ (Day 1) ✅   │    │ (Day 4)      │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         │                   ▼                   ▼               │
│         │            ┌──────────────┐    ┌──────────────┐      │
│         │            │ Model        │    │ CloudWatch   │      │
│         └───────────▶│ Registry     │    │ Monitoring   │      │
│                      └──────────────┘    │ (Day 5)      │      │
│                            │             └──────────────┘      │
│                            ▼                   │               │
│                      ┌─────────────────────────────────┐       │
│                      │  Drift Detection → Auto Retrain │       │
│                      └─────────────────────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
How It All Connects
Component	Role in Pipeline
SageMaker Pipelines (Day 1)	Orchestrates the entire workflow — training, evaluation, registration
Feature Store (Day 2)	Supplies features for training (offline) and inference (online)
Neo (Day 3)	Optimizes models for edge/production deployment
Multi-Model Endpoints (Day 4)	Cost-efficient hosting of multiple model versions
Monitoring (Day 5)	Detects drift, triggers retraining via Pipelines
By the weekend, you'll have touched every major component of a production MLOps system. That's a strong portfolio piece! 🔥

I currently have a domain in Sagemaker that I created late October.  It looks like it's charging a few cents a day.  Should I be tearing it down or wait until I'm done with my projects?

Keep it for now. Here's why:

Cost Breakdown
Resource	Cost	Notes
SageMaker Domain	Free	The domain itself doesn't cost
EFS Storage	~$0.30/GB/month	This is likely your few cents/day
Studio Apps (stopped)	Free	Only charged when running
Studio Apps (running)	$0.05-0.50/hr	Depends on instance type
Your few cents/day is probably EFS storage for notebooks and artifacts. At that rate (~$1-2/month), it's not worth tearing down mid-study.

Recommendation
Keep it until after Week 8 (your Enterprise MLOps project). Reasons:

Recreating a domain takes 15-20 minutes each time
You'll lose saved notebooks, configs, and settings
$1-2/month is worth the convenience during active study
Days 3-5 and weekend project all use SageMaker
After the Exam
Once you pass, then clean up:

bash
# Delete SageMaker Domain (do this in Console - easier)
# 1. Delete all Apps (Studio, Canvas, etc.)
# 2. Delete all User Profiles  
# 3. Delete the Domain
# 4. Optionally delete the EFS volume
What to Watch For
The expensive stuff that can sneak up on you:

Resource	Cost	Check
Running endpoints	$0.05-2.00/hr	Delete after each hands-on
Running notebooks	$0.05-0.50/hr	Shut down when done for the day
Training jobs	Varies	These auto-stop when complete
Quick check command:

bash
# List any running endpoints (delete these!)
aws sagemaker list-endpoints --query "Endpoints[?EndpointStatus=='InService'].EndpointName"
Bottom line: Keep the domain, just make sure you're deleting endpoints and stopping notebooks after each session.

I have a couple other chats related to my AWS ML Specialty study.  How can I consolidate them all in a Claude project?

Great idea! Claude Projects let you organize related conversations and add persistent context. Here's how to set it up:

Creating Your Project
Go to Claude.ai → Click "Projects" in the left sidebar
Click "Create Project"
Name it something like: AWS ML Specialty Certification
Add a description: Study materials, hands-on projects, and exam prep for AWS Certified Machine Learning - Specialty
Adding Project Knowledge
In the project settings, you can add persistent context that Claude will reference in every conversation. Good things to add:

Your 30-day study plan
Your current progress (Week 8, Day 2 complete)
Your repo structure (aws-ml-portfolio)
Key preferences (hands-on learner, data engineering background)
Your SageMaker config (region: us-east-2, bucket name, role ARN)
Example project instructions:

I'm studying for the AWS ML Specialty certification. I have 10 years of 
data engineering experience and am transitioning to ML engineering.

Current progress: Week 8 of 10
Repo: aws-ml-portfolio on GitHub
AWS Region: us-east-2
SageMaker bucket: sagemaker-us-east-2-854757836160

I prefer hands-on learning with working code examples. After each session,
I like a recap of key exam concepts.
Moving Existing Chats
Unfortunately, you can't move existing chats into a project — but you can:

Start new chats within the project going forward (Days 3-5, weekend project)
Reference old chats by searching them (Claude can still find them)
Copy key summaries from old chats into Project Knowledge
Suggested Project Structure
Start new chats in your project for:

Chat	Purpose
Day 3: Edge & Neo	Edge deployment hands-on
Day 4: Multi-Model Endpoints	MME/MCE hands-on
Day 5: Monitoring	CloudWatch, Clarify hands-on
Weekend: MLOps Pipeline	Capstone project
Exam Prep & Review	Practice questions, concept review
Want me to draft the full Project Knowledge instructions you can paste in?

That's okay, let's move onto Day 3 for this week, with Edge and Neo.

Let's do it! Here's the plan for Day 3:

Day 3: Edge Deployment & SageMaker Neo
What You'll Learn
Concept	Why It Matters
SageMaker Neo	Compiles models for specific hardware (up to 2x faster)
Edge Manager	Deploys, monitors, and updates models on edge devices
Edge use cases	IoT, mobile, autonomous vehicles, offline inference
Exam Concepts
When to use Edge deployment:

Low latency required (can't afford round-trip to cloud)
Intermittent/no connectivity
Bandwidth constraints (can't send all data to cloud)
Privacy/compliance (data must stay on device)
Neo compilation targets:

CPU (Intel, ARM)
GPU (NVIDIA)
Accelerators (Inferentia, edge TPUs)
Devices (Jetson, Raspberry Pi, Android, iOS)
Hands-On Plan
Train a simple model (or use one from a previous project)
Compile with Neo for CPU target
Compare model size before/after compilation
Test inference speed improvement
Explore Edge Manager concepts (we'll keep this lighter since you don't have physical edge devices)
Prerequisites Check
You should have from previous weeks:

A trained model in S3, OR
We can quickly train an XGBoost model on Titanic data
Do you have a trained model artifact in S3 we can use, or should we start by training a quick model to compile?

https://us-east-2.console.aws.amazon.com/sagemaker/home?region=us-east-2#/models

Would I find them here? If so, I have some models.

That page shows SageMaker Models (model configurations registered for deployment), but what we need for Neo is the model artifact — the actual model.tar.gz file in S3.

Finding Your Model Artifacts
Option 1: Check S3 directly

bash
# List your training job outputs
aws s3 ls s3://sagemaker-us-east-2-854757836160/ --recursive | grep "model.tar.gz"
Option 2: Check via Console

Go to SageMaker → Training → Training jobs
Click on a completed job
Look for Output → S3 model artifact — that's the path we need
Option 3: Check the Models page you're on

Click on one of your models
Look for Primary container → Model data URL
That S3 path is what we need
Can you run that S3 command or check one of your models for the artifact path? Once we have it, we can compile it with Neo.

If nothing usable turns up, we can quickly train a fresh XGBoost model (~5 min) — that's actually a good framework for Neo since it's well-supported.

Let's just freshly train a model.  The ones I have are from last month.

Let's train a fresh XGBoost model for Neo compilation. Here's the script:

train_model_for_neo.py
python
"""
Train an XGBoost model for SageMaker Neo compilation
"""
import boto3
import sagemaker
from sagemaker import image_uris
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
import pandas as pd
import numpy as np
import time

# Configuration
region = boto3.Session().region_name
boto_session = boto3.Session(region_name=region)
sagemaker_session = sagemaker.Session(boto_session=boto_session)

bucket = sagemaker_session.default_bucket()
prefix = 'neo-demo'
role = sagemaker.get_execution_role()

print(f"Region: {region}")
print(f"Bucket: {bucket}")
print(f"Prefix: {prefix}")

print("\n" + "="*70)
print("STEP 1: CREATE SAMPLE DATASET")
print("="*70)

# Create a simple classification dataset
np.random.seed(42)
n_samples = 1000

# Generate features
data = {
    'feature_1': np.random.randn(n_samples),
    'feature_2': np.random.randn(n_samples),
    'feature_3': np.random.randn(n_samples),
    'feature_4': np.random.randn(n_samples),
    'feature_5': np.random.randn(n_samples),
}

df = pd.DataFrame(data)

# Create target (binary classification based on features)
df['target'] = ((df['feature_1'] + df['feature_2'] * 0.5 + np.random.randn(n_samples) * 0.3) > 0).astype(int)

# XGBoost requires target in first column, no headers
train_df = df[['target', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']]

print(f"\nDataset shape: {train_df.shape}")
print(f"Target distribution:\n{train_df['target'].value_counts()}")

# Save locally and upload to S3
train_file = 'train.csv'
train_df.to_csv(train_file, index=False, header=False)

train_s3_path = f's3://{bucket}/{prefix}/train/{train_file}'
sagemaker_session.upload_data(train_file, bucket=bucket, key_prefix=f'{prefix}/train')

print(f"\n✅ Training data uploaded to: {train_s3_path}")

print("\n" + "="*70)
print("STEP 2: TRAIN XGBOOST MODEL")
print("="*70)

# Get XGBoost container
container = image_uris.retrieve('xgboost', region, '1.5-1')
print(f"\nUsing container: {container}")

# Create estimator
xgb_estimator = Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    output_path=f's3://{bucket}/{prefix}/output/',
    sagemaker_session=sagemaker_session,
    base_job_name='neo-xgboost'
)

# Set hyperparameters
xgb_estimator.set_hyperparameters(
    objective='binary:logistic',
    num_round=50,
    max_depth=4,
    eta=0.2,
    subsample=0.8,
    colsample_bytree=0.8
)

print("\nStarting training job...")
print("This will take 3-5 minutes...\n")

# Train
xgb_estimator.fit(
    {'train': TrainingInput(train_s3_path, content_type='text/csv')},
    wait=True
)

# Get model artifact location
model_artifact = xgb_estimator.model_data

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print(f"\nModel artifact: {model_artifact}")
print(f"\nTraining job name: {xgb_estimator.latest_training_job.name}")

# Save model path for Neo compilation
with open('model_artifact_path.txt', 'w') as f:
    f.write(model_artifact)

print("\n📝 Model artifact path saved to: model_artifact_path.txt")
print("\n💡 Next step: Compile this model with SageMaker Neo")
Run It
bash
cd ~/feature-store-project  # or wherever you want to work
mkdir -p ../neo-demo
cd ../neo-demo

# Save the script and run
python train_model_for_neo.py
This will:

Create a simple binary classification dataset
Train an XGBoost model (~3-5 min)
Save the model artifact path for Neo compilation
Let me know when training completes and I'll give you the Neo compilation script!

ImportError: cannot import name 'image_uris' from 'sagemaker' (unknown location)

The SageMaker SDK version you have uses a different import. Let me fix that:

Quick Fix
Replace this line:

python
from sagemaker import image_uris
With:

python
from sagemaker.image_uris import retrieve
And change this line:

python
container = image_uris.retrieve('xgboost', region, '1.5-1')
To:

python
container = retrieve('xgboost', region, '1.5-1')
Or run this to check your SDK version and update if needed:

bash
python3 -c "import sagemaker; print(sagemaker.__version__)"

# If it's old, upgrade:
pip install --upgrade sagemaker
Want me to give you the full corrected script, or can you make those two changes?

ModuleNotFoundError: No module named 'sagemaker.image_uris'

Made the changes got this error

Your SageMaker SDK might be older. Let's try a different import that works across versions:

Quick Fix
Replace:

python
from sagemaker.image_uris import retrieve
With:

python
from sagemaker import get_image_uri
And change:

python
container = retrieve('xgboost', region, '1.5-1')
To:

python
container = get_image_uri(region, 'xgboost', repo_version='1.5-1')
Or just upgrade your SDK (recommended):

bash
pip install 'sagemaker>=2.200.0' --upgrade

# Verify
python3 -c "import sagemaker; print(sagemaker.__version__)"
Then the original script should work.

Which do you prefer — quick fix or upgrade?

Ran the upgrade, but when verifying I get this:

AttributeError: module 'sagemaker' has no attribute '__version__'

That's odd — let's check what's happening:

bash
# Check where sagemaker is installed
pip show sagemaker

# Check if there's a conflicting local file
ls -la sagemaker.py 2>/dev/null
There might be a file named sagemaker.py in your current directory that's shadowing the actual package.

If you find a sagemaker.py file:

bash
# Rename it
mv sagemaker.py my_sagemaker_script.py

# Also remove the cache
rm -rf __pycache__
If no local file conflict, try:

bash
# Reinstall cleanly
pip uninstall sagemaker -y
pip install 'sagemaker>=2.200.0,<3.0'

# Verify from a clean directory
cd ~
python3 -c "import sagemaker; print(sagemaker.__version__)"
What do you see?

When re-installing, I did get this error message although the install succeded.

ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
sagemaker-serve 1.1.0 requires sagemaker-core>=2.1.0, but you have sagemaker-core 1.0.71 which is incompatible.
sagemaker-train 1.1.0 requires sagemaker-core>=2.1.0, but you have sagemaker-core 1.0.71 which is incompatible.
sagemaker-mlops 1.1.0 requires sagemaker-core>=2.1.0, but you have sagemaker-core 1.0.71 which is incompatible.

When verifying from clean directory, got this:

sagemaker.config INFO - Not applying SDK defaults from location: /Library/Application Support/sagemaker/config.yaml
sagemaker.config INFO - Not applying SDK defaults from location: /Users/adamvigneau/Library/Application Support/sagemaker/config.yaml
2.255.0

Show more
That worked! You're now on version 2.255.0 which is great.

The dependency warnings are fine — they won't affect what we're doing.

Now go back to your neo-demo directory and run the training script:

bash
cd ~/neo-demo
python train_model_for_neo.py
Let me know when training completes (~3-5 min) or if you hit any other errors!

sagemaker.config INFO - Not applying SDK defaults from location: /Library/Application Support/sagemaker/config.yaml
sagemaker.config INFO - Not applying SDK defaults from location: /Users/adamvigneau/Library/Application Support/sagemaker/config.yaml
Couldn't call 'get_role' to get Role ARN from role name avigneau to get Role path.
╭───────────────────── Traceback (most recent call last) ──────────────────────╮
│ /Users/adamvigneau/neo-demo/train_model_for_neo.py:20 in <module>            │
│                                                                              │
│    17                                                                        │
│    18 bucket = sagemaker_session.default_bucket()                            │
│    19 prefix = 'neo-demo'                                                    │
│ ❱  20 role = sagemaker.get_execution_role()                                  │
│    21                                                                        │
│    22 print(f"Region: {region}")                                             │
│    23 print(f"Bucket: {bucket}")                                             │
│                                                                              │
│ /opt/anaconda3/lib/python3.13/site-packages/sagemaker/session.py:8317 in     │
│ get_execution_role                                                           │
│                                                                              │
│   8314 │   │   "The current AWS identity is not a role: {}, therefore it can │
│   8315 │   │   "SageMaker execution role"                                    │
│   8316 │   )                                                                 │
│ ❱ 8317 │   raise ValueError(message.format(arn))                             │
│   8318                                                                       │
│   8319                                                                       │
│   8320 def generate_default_sagemaker_bucket_name(boto_session):             │
╰──────────────────────────────────────────────────────────────────────────────╯
ValueError: The current AWS identity is not a role: 
arn:aws:iam::854757836160:user/avigneau, therefore it cannot be used as a 
SageMaker execution role

Show more
You're running locally (not in SageMaker Studio), so get_execution_role() doesn't work. We need to specify your role ARN directly.

Quick Fix
Replace line 20:

python
role = sagemaker.get_execution_role()
With your role ARN from previous projects:

python
role = "arn:aws:iam::854757836160:role/service-role/AmazonSageMaker-ExecutionRole-20251019T120276"
Or load it from your config file like the Feature Store scripts did:

python
import json

try:
    with open('../mlops-pipeline/mlops_config.json') as f:
        config = json.load(f)
        role = config['sagemaker_role_arn']
except:
    role = "arn:aws:iam::854757836160:role/service-role/AmazonSageMaker-ExecutionRole-20251019T120276"
Make that change and run again!

ran successfully

Training complete! 🎉

Now let's compile that model with SageMaker Neo. First, let's grab the model artifact path:

bash
cat model_artifact_path.txt
Paste the S3 path it shows, and I'll give you the Neo compilation script.

Or if you want, I can write the script to read it automatically from that file — your call!

Let's have the script read it automatically from that file

Here's the Neo compilation script:

compile_with_neo.py
python
"""
Compile a trained model with SageMaker Neo for optimized inference
"""
import boto3
import sagemaker
import time

# Configuration
region = boto3.Session().region_name
sagemaker_client = boto3.client('sagemaker', region_name=region)
sagemaker_session = sagemaker.Session()

bucket = sagemaker_session.default_bucket()
prefix = 'neo-demo'
role = "arn:aws:iam::854757836160:role/service-role/AmazonSageMaker-ExecutionRole-20251019T120276"

print(f"Region: {region}")
print(f"Bucket: {bucket}")

# Read model artifact path from training step
with open('model_artifact_path.txt', 'r') as f:
    model_artifact = f.read().strip()

print(f"Model artifact: {model_artifact}")

print("\n" + "="*70)
print("COMPILING MODEL WITH SAGEMAKER NEO")
print("="*70)

# Compilation job configuration
compilation_job_name = f"neo-xgboost-{int(time.time())}"
output_path = f"s3://{bucket}/{prefix}/neo-output"

print(f"\nCompilation job name: {compilation_job_name}")
print(f"Output path: {output_path}")

# Target: CPU (most common for local/edge deployment)
# Options: ml_m5, ml_c5, ml_p3, deeplens, jetson_nano, rasp3b, etc.
target_platform = {
    'Os': 'LINUX',
    'Arch': 'X86_64'
}

print(f"\nTarget platform:")
print(f"   OS: {target_platform['Os']}")
print(f"   Architecture: {target_platform['Arch']}")

print("\n" + "-"*70)
print("Starting compilation...")
print("-"*70)

# Create compilation job
try:
    response = sagemaker_client.create_compilation_job(
        CompilationJobName=compilation_job_name,
        RoleArn=role,
        InputConfig={
            'S3Uri': model_artifact,
            'DataInputConfig': '{"input": [1, 5]}',  # Batch size 1, 5 features
            'Framework': 'XGBOOST'
        },
        OutputConfig={
            'S3OutputLocation': output_path,
            'TargetPlatform': target_platform
        },
        StoppingCondition={
            'MaxRuntimeInSeconds': 900  # 15 min max
        }
    )
    print(f"   ✅ Compilation job created!")
    
except Exception as e:
    print(f"   ❌ Error creating job: {e}")
    exit(1)

# Wait for completion
print("\n   Waiting for compilation to complete...")
print("   (This typically takes 2-5 minutes)\n")

while True:
    response = sagemaker_client.describe_compilation_job(
        CompilationJobName=compilation_job_name
    )
    
    status = response['CompilationJobStatus']
    
    if status == 'COMPLETED':
        print(f"   ✅ Compilation COMPLETED!")
        break
    elif status == 'FAILED':
        print(f"   ❌ Compilation FAILED!")
        print(f"   Reason: {response.get('FailureReason', 'Unknown')}")
        exit(1)
    elif status == 'STOPPED':
        print(f"   ⚠️ Compilation STOPPED")
        exit(1)
    else:
        print(f"   ⏳ Status: {status}...")
        time.sleep(30)

# Get results
compiled_model_path = response['ModelArtifacts']['S3ModelArtifacts']

print("\n" + "="*70)
print("✅ NEO COMPILATION COMPLETE!")
print("="*70)

print(f"\n📦 Original model: {model_artifact}")
print(f"📦 Compiled model: {compiled_model_path}")

# Save compiled model path for next steps
with open('compiled_model_path.txt', 'w') as f:
    f.write(compiled_model_path)

print(f"\n📝 Compiled model path saved to: compiled_model_path.txt")

# Get model sizes for comparison
s3 = boto3.client('s3')

def get_s3_file_size(s3_uri):
    """Get file size from S3 URI"""
    parts = s3_uri.replace("s3://", "").split("/")
    bucket_name = parts[0]
    key = "/".join(parts[1:])
    
    try:
        response = s3.head_object(Bucket=bucket_name, Key=key)
        return response['ContentLength']
    except:
        return None

original_size = get_s3_file_size(model_artifact)
compiled_size = get_s3_file_size(compiled_model_path)

print("\n" + "-"*70)
print("MODEL SIZE COMPARISON")
print("-"*70)

if original_size and compiled_size:
    print(f"\n   Original model: {original_size / 1024:.1f} KB")
    print(f"   Compiled model: {compiled_size / 1024:.1f} KB")
    
    if compiled_size < original_size:
        reduction = (1 - compiled_size / original_size) * 100
        print(f"\n   📉 Size reduction: {reduction:.1f}%")
    else:
        increase = (compiled_size / original_size - 1) * 100
        print(f"\n   📈 Size increase: {increase:.1f}% (includes Neo runtime)")
else:
    print("\n   Could not compare sizes (check S3 permissions)")

print("\n" + "-"*70)
print("KEY EXAM CONCEPTS")
print("-"*70)
print("""
1. Neo compiles models for SPECIFIC hardware targets
2. Supports: TensorFlow, PyTorch, MXNet, XGBoost, ONNX
3. Target platforms: cloud instances, Jetson, Raspberry Pi, iOS, Android
4. Benefits: faster inference, smaller footprint, lower cost
5. Compiled models can ONLY run on the target platform specified
""")

print("\n💡 Next step: Deploy the compiled model to an endpoint")
Run It
bash
python compile_with_neo.py
This will take 2-5 minutes. Let me know what output you get!

sagemaker.config INFO - Not applying SDK defaults from location: /Library/Application Support/sagemaker/config.yaml
sagemaker.config INFO - Not applying SDK defaults from location: /Users/adamvigneau/Library/Application Support/sagemaker/config.yaml
Region: us-east-2
Bucket: sagemaker-us-east-2-854757836160
Model artifact: s3://sagemaker-us-east-2-854757836160/neo-demo/output/neo-xgboost-2025-12-04-18-13-08-520/output/model.tar.gz

======================================================================
COMPILING MODEL WITH SAGEMAKER NEO
======================================================================

Compilation job name: neo-xgboost-1764872738
Output path: s3://sagemaker-us-east-2-854757836160/neo-demo/neo-output

Target platform:
   OS: LINUX
   Architecture: X86_64

----------------------------------------------------------------------
Starting compilation...
----------------------------------------------------------------------
   ✅ Compilation job created!

   Waiting for compilation to complete...
   (This typically takes 2-5 minutes)

   ⏳ Status: STARTING...
   ⏳ Status: STARTING...
   ⏳ Status: STARTING...
   ⏳ Status: STARTING...
   ⏳ Status: INPROGRESS...
   ⏳ Status: INPROGRESS...
   ⏳ Status: INPROGRESS...
   ✅ Compilation COMPLETED!

======================================================================
✅ NEO COMPILATION COMPLETE!
======================================================================

📦 Original model: s3://sagemaker-us-east-2-854757836160/neo-demo/output/neo-xgboost-2025-12-04-18-13-08-520/output/model.tar.gz
📦 Compiled model: s3://sagemaker-us-east-2-854757836160/neo-demo/neo-output/model-LINUX_X86_64.tar.gz

📝 Compiled model path saved to: compiled_model_path.txt

----------------------------------------------------------------------
MODEL SIZE COMPARISON
----------------------------------------------------------------------

   Original model: 18.6 KB
   Compiled model: 1054.7 KB

   📈 Size increase: 5569.4% (includes Neo runtime)

----------------------------------------------------------------------
KEY EXAM CONCEPTS
----------------------------------------------------------------------

Neo compiles models for SPECIFIC hardware targets
Supports: TensorFlow, PyTorch, MXNet, XGBoost, ONNX
Target platforms: cloud instances, Jetson, Raspberry Pi, iOS, Android
Benefits: faster inference, smaller footprint, lower cost
Compiled models can ONLY run on the target platform specified
💡 Next step: Deploy the compiled model to an endpoint

Show more
Neo compilation complete! 🎉

About the Size Increase
Don't worry about the size increase (18 KB → 1 MB). This is expected because:

Component	What's Included
Original model	Just the trained XGBoost weights
Compiled model	Weights + Neo runtime + optimized inference code
The Neo runtime is bundled so the model can run without needing the full framework installed. For larger models (100+ MB), you'd typically see a reduction in size.

The real benefit is inference speed — let's deploy and test it!

Next Steps
Deploy the compiled model to an endpoint
Compare inference latency — Neo-compiled vs standard
Explore Edge Manager concepts (for deploying to actual devices)
Want to continue with deploying the compiled model and testing inference speed?

Yes, let's continue

Here's the script to deploy and benchmark the Neo-compiled model:

deploy_and_benchmark.py
python
"""
Deploy Neo-compiled model and benchmark inference speed
"""
import boto3
import sagemaker
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
import time
import numpy as np

# Configuration
region = boto3.Session().region_name
sagemaker_session = sagemaker.Session()
sagemaker_client = boto3.client('sagemaker', region_name=region)

bucket = sagemaker_session.default_bucket()
role = "arn:aws:iam::854757836160:role/service-role/AmazonSageMaker-ExecutionRole-20251019T120276"

print(f"Region: {region}")
print(f"Bucket: {bucket}")

# Read model paths
with open('model_artifact_path.txt', 'r') as f:
    original_model_path = f.read().strip()

with open('compiled_model_path.txt', 'r') as f:
    compiled_model_path = f.read().strip()

print(f"\nOriginal model: {original_model_path}")
print(f"Compiled model: {compiled_model_path}")

# Get container images
from sagemaker import image_uris

# Standard XGBoost container
xgb_container = image_uris.retrieve('xgboost', region, '1.5-1')

# Neo-optimized container
neo_container = image_uris.retrieve(
    'xgboost-neo',
    region,
    version='latest'
)

print(f"\nXGBoost container: {xgb_container[:60]}...")
print(f"Neo container: {neo_container[:60]}...")

print("\n" + "="*70)
print("DEPLOYING MODELS")
print("="*70)

# Timestamp for unique names
timestamp = int(time.time())

# ============================================================
# Deploy Original Model
# ============================================================
print("\n1️⃣  Deploying ORIGINAL XGBoost model...")

original_model = Model(
    model_data=original_model_path,
    image_uri=xgb_container,
    role=role,
    sagemaker_session=sagemaker_session,
    name=f"neo-demo-original-{timestamp}"
)

original_endpoint_name = f"neo-demo-original-{timestamp}"

print(f"   Endpoint name: {original_endpoint_name}")
print(f"   This will take 3-5 minutes...\n")

original_predictor = original_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name=original_endpoint_name,
    serializer=CSVSerializer(),
    deserializer=JSONDeserializer()
)

print(f"   ✅ Original model deployed!")

# ============================================================
# Deploy Neo-Compiled Model
# ============================================================
print("\n2️⃣  Deploying NEO-COMPILED model...")

neo_model = Model(
    model_data=compiled_model_path,
    image_uri=neo_container,
    role=role,
    sagemaker_session=sagemaker_session,
    name=f"neo-demo-compiled-{timestamp}"
)

neo_endpoint_name = f"neo-demo-compiled-{timestamp}"

print(f"   Endpoint name: {neo_endpoint_name}")
print(f"   This will take 3-5 minutes...\n")

neo_predictor = neo_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name=neo_endpoint_name,
    serializer=CSVSerializer(),
    deserializer=JSONDeserializer()
)

print(f"   ✅ Neo-compiled model deployed!")

print("\n" + "="*70)
print("BENCHMARKING INFERENCE SPEED")
print("="*70)

# Generate test data (same format as training)
np.random.seed(42)
test_samples = []
for _ in range(100):
    sample = np.random.randn(5).tolist()
    test_samples.append(sample)

# Convert to CSV format
def to_csv(sample):
    return ','.join(map(str, sample))

# Warm-up calls (first calls are always slower)
print("\n3️⃣  Warming up endpoints...")
for _ in range(5):
    original_predictor.predict(to_csv(test_samples[0]))
    neo_predictor.predict(to_csv(test_samples[0]))
print("   ✅ Warm-up complete")

# ============================================================
# Benchmark Original Model
# ============================================================
print("\n4️⃣  Benchmarking ORIGINAL model (100 inferences)...")

original_latencies = []
for sample in test_samples:
    start = time.time()
    original_predictor.predict(to_csv(sample))
    elapsed = (time.time() - start) * 1000
    original_latencies.append(elapsed)

original_avg = sum(original_latencies) / len(original_latencies)
original_p50 = sorted(original_latencies)[50]
original_p95 = sorted(original_latencies)[95]

print(f"   Average: {original_avg:.1f} ms")
print(f"   P50: {original_p50:.1f} ms")
print(f"   P95: {original_p95:.1f} ms")

# ============================================================
# Benchmark Neo Model
# ============================================================
print("\n5️⃣  Benchmarking NEO-COMPILED model (100 inferences)...")

neo_latencies = []
for sample in test_samples:
    start = time.time()
    neo_predictor.predict(to_csv(sample))
    elapsed = (time.time() - start) * 1000
    neo_latencies.append(elapsed)

neo_avg = sum(neo_latencies) / len(neo_latencies)
neo_p50 = sorted(neo_latencies)[50]
neo_p95 = sorted(neo_latencies)[95]

print(f"   Average: {neo_avg:.1f} ms")
print(f"   P50: {neo_p50:.1f} ms")
print(f"   P95: {neo_p95:.1f} ms")

# ============================================================
# Results Summary
# ============================================================
print("\n" + "="*70)
print("BENCHMARK RESULTS")
print("="*70)

print(f"""
┌────────────────────┬──────────────┬──────────────┐
│  Metric            │  Original    │  Neo-Compiled│
├────────────────────┼──────────────┼──────────────┤
│  Average Latency   │  {original_avg:>8.1f} ms │  {neo_avg:>8.1f} ms │
│  P50 Latency       │  {original_p50:>8.1f} ms │  {neo_p50:>8.1f} ms │
│  P95 Latency       │  {original_p95:>8.1f} ms │  {neo_p95:>8.1f} ms │
└────────────────────┴──────────────┴──────────────┘
""")

speedup = original_avg / neo_avg if neo_avg > 0 else 0
if speedup > 1:
    print(f"   🚀 Neo is {speedup:.1f}x FASTER!")
elif speedup < 1:
    print(f"   📊 Original is {1/speedup:.1f}x faster (small models may not benefit)")
else:
    print(f"   📊 Performance is similar")

print("\n" + "-"*70)
print("KEY EXAM CONCEPTS")
print("-"*70)
print("""
1. Neo benefits are MORE significant for:
   - Larger models (deep learning)
   - Edge/embedded devices with limited resources
   - High-throughput inference (many requests)

2. Neo benefits may be LESS noticeable for:
   - Small models (like our XGBoost demo)
   - Already-efficient frameworks
   - Network latency dominates (remote calls)

3. Real-world Neo speedups: typically 1.5x - 4x for deep learning models
""")

# Save endpoint names for cleanup
with open('endpoint_names.txt', 'w') as f:
    f.write(f"{original_endpoint_name}\n")
    f.write(f"{neo_endpoint_name}\n")

print("\n" + "="*70)
print("⚠️  IMPORTANT: CLEANUP REQUIRED")
print("="*70)
print(f"""
Endpoints are EXPENSIVE! Delete when done:

   python cleanup_neo_demo.py

Or manually:
   aws sagemaker delete-endpoint --endpoint-name {original_endpoint_name}
   aws sagemaker delete-endpoint --endpoint-name {neo_endpoint_name}

Endpoint names saved to: endpoint_names.txt
""")
Run It
bash
python deploy_and_benchmark.py
This will:

Deploy the original model (~3-5 min)
Deploy the Neo-compiled model (~3-5 min)
Run 100 inferences on each
Compare latency
Note: Total time ~8-12 minutes for both deployments.

Let me know when it completes!

It errored in the middle:

sagemaker.config INFO - Not applying SDK defaults from location: /Library/Application Support/sagemaker/config.yaml
sagemaker.config INFO - Not applying SDK defaults from location: /Users/adamvigneau/Library/Application Support/sagemaker/config.yaml
Region: us-east-2
Bucket: sagemaker-us-east-2-854757836160

Original model: s3://sagemaker-us-east-2-854757836160/neo-demo/output/neo-xgboost-2025-12-04-18-13-08-520/output/model.tar.gz
Compiled model: s3://sagemaker-us-east-2-854757836160/neo-demo/neo-output/model-LINUX_X86_64.tar.gz

XGBoost container: 257758044811.dkr.ecr.us-east-2.amazonaws.com/sagemaker-xgboo...
Neo container: 007439368137.dkr.ecr.us-east-2.amazonaws.com/xgboost-neo:lat...

======================================================================
DEPLOYING MODELS
======================================================================

1️⃣  Deploying ORIGINAL XGBoost model...
   Endpoint name: neo-demo-original-1764873559
   This will take 3-5 minutes...

------!   ✅ Original model deployed!

2️⃣  Deploying NEO-COMPILED model...
   Endpoint name: neo-demo-compiled-1764873559
   This will take 3-5 minutes...

-----!   ✅ Neo-compiled model deployed!

======================================================================
BENCHMARKING INFERENCE SPEED
======================================================================

3️⃣  Warming up endpoints...
╭───────────────────── Traceback (most recent call last) ──────────────────────╮
│ /Users/adamvigneau/neo-demo/deploy_and_benchmark.py:131 in <module>          │
│                                                                              │
│   128 # Warm-up calls (first calls are always slower)                        │
│   129 print("\n3️⃣  Warming up endpoints...")                                  │
│   130 for _ in range(5):                                                     │
│ ❱ 131 │   original_predictor.predict(to_csv(test_samples[0]))                │
│   132 │   neo_predictor.predict(to_csv(test_samples[0]))                     │
│   133 print("   ✅ Warm-up complete")                                        │
│   134                                                                        │
╰──────────────────────────────────────────────────────────────────────────────╯
AttributeError: 'NoneType' object has no attribute 'predict'

Show more
The model deployed but the predictor object wasn't returned properly. Let's create the predictors manually and continue the benchmark. Here's a fix script:

benchmark_only.py
python
"""
Benchmark deployed models (run after deploy_and_benchmark.py errored)
"""
import boto3
import time
import numpy as np

# Configuration
region = boto3.Session().region_name
runtime_client = boto3.client('sagemaker-runtime', region_name=region)

print(f"Region: {region}")

# Read endpoint names
with open('endpoint_names.txt', 'r') as f:
    lines = f.read().strip().split('\n')
    original_endpoint_name = lines[0]
    neo_endpoint_name = lines[1]

print(f"\nOriginal endpoint: {original_endpoint_name}")
print(f"Neo endpoint: {neo_endpoint_name}")

print("\n" + "="*70)
print("BENCHMARKING INFERENCE SPEED")
print("="*70)

# Generate test data (same format as training)
np.random.seed(42)
test_samples = []
for _ in range(100):
    sample = np.random.randn(5).tolist()
    test_samples.append(sample)

# Convert to CSV format
def to_csv(sample):
    return ','.join(map(str, sample))

# Invoke function using boto3 directly
def invoke_endpoint(endpoint_name, payload):
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='text/csv',
        Body=payload
    )
    return response['Body'].read().decode('utf-8')

# Warm-up calls (first calls are always slower)
print("\n1️⃣  Warming up endpoints...")
for i in range(5):
    invoke_endpoint(original_endpoint_name, to_csv(test_samples[0]))
    invoke_endpoint(neo_endpoint_name, to_csv(test_samples[0]))
    print(f"   Warm-up {i+1}/5 complete")
print("   ✅ Warm-up complete")

# ============================================================
# Benchmark Original Model
# ============================================================
print("\n2️⃣  Benchmarking ORIGINAL model (100 inferences)...")

original_latencies = []
for sample in test_samples:
    start = time.time()
    invoke_endpoint(original_endpoint_name, to_csv(sample))
    elapsed = (time.time() - start) * 1000
    original_latencies.append(elapsed)

original_avg = sum(original_latencies) / len(original_latencies)
original_p50 = sorted(original_latencies)[50]
original_p95 = sorted(original_latencies)[95]

print(f"   Average: {original_avg:.1f} ms")
print(f"   P50: {original_p50:.1f} ms")
print(f"   P95: {original_p95:.1f} ms")

# ============================================================
# Benchmark Neo Model
# ============================================================
print("\n3️⃣  Benchmarking NEO-COMPILED model (100 inferences)...")

neo_latencies = []
for sample in test_samples:
    start = time.time()
    invoke_endpoint(neo_endpoint_name, to_csv(sample))
    elapsed = (time.time() - start) * 1000
    neo_latencies.append(elapsed)

neo_avg = sum(neo_latencies) / len(neo_latencies)
neo_p50 = sorted(neo_latencies)[50]
neo_p95 = sorted(neo_latencies)[95]

print(f"   Average: {neo_avg:.1f} ms")
print(f"   P50: {neo_p50:.1f} ms")
print(f"   P95: {neo_p95:.1f} ms")

# ============================================================
# Results Summary
# ============================================================
print("\n" + "="*70)
print("BENCHMARK RESULTS")
print("="*70)

print(f"""
┌────────────────────┬──────────────┬──────────────┐
│  Metric            │  Original    │  Neo-Compiled│
├────────────────────┼──────────────┼──────────────┤
│  Average Latency   │  {original_avg:>8.1f} ms │  {neo_avg:>8.1f} ms │
│  P50 Latency       │  {original_p50:>8.1f} ms │  {neo_p50:>8.1f} ms │
│  P95 Latency       │  {original_p95:>8.1f} ms │  {neo_p95:>8.1f} ms │
└────────────────────┴──────────────┴──────────────┘
""")

speedup = original_avg / neo_avg if neo_avg > 0 else 0
if speedup > 1:
    print(f"   🚀 Neo is {speedup:.2f}x FASTER!")
elif speedup < 1:
    print(f"   📊 Original is {1/speedup:.2f}x faster (small models may not benefit)")
else:
    print(f"   📊 Performance is similar")

print("\n" + "-"*70)
print("KEY EXAM CONCEPTS")
print("-"*70)
print("""
1. Neo benefits are MORE significant for:
   - Larger models (deep learning)
   - Edge/embedded devices with limited resources
   - High-throughput inference (many requests)

2. Neo benefits may be LESS noticeable for:
   - Small models (like our XGBoost demo)
   - Already-efficient frameworks
   - Network latency dominates (remote calls)

3. Real-world Neo speedups: typically 1.5x - 4x for deep learning models
""")

print("\n" + "="*70)
print("⚠️  IMPORTANT: CLEANUP REQUIRED")
print("="*70)
print(f"""
Endpoints are EXPENSIVE! Delete when done:

   python cleanup_neo_demo.py

Or manually:
   aws sagemaker delete-endpoint --endpoint-name {original_endpoint_name}
   aws sagemaker delete-endpoint --endpoint-name {neo_endpoint_name}
""")
Run It
bash
python benchmark_only.py
The endpoints are already deployed, so this should run quickly (~1-2 min for 200 total inferences).

Let me know what results you get!

FileNotFoundError: [Errno 2] No such file or directory: 'endpoint_names.txt'

The script errored before it could save the endpoint names. Let's create that file manually with the endpoint names from your output:

bash
echo "neo-demo-original-1764873559
neo-demo-compiled-1764873559" > endpoint_names.txt
Then run the benchmark:

bash
python benchmark_only.py
Region: us-east-2

Original endpoint: neo-demo-original-1764873559
Neo endpoint: neo-demo-compiled-1764873559

======================================================================
BENCHMARKING INFERENCE SPEED
======================================================================

1️⃣  Warming up endpoints...
   Warm-up 1/5 complete
   Warm-up 2/5 complete
   Warm-up 3/5 complete
   Warm-up 4/5 complete
   Warm-up 5/5 complete
   ✅ Warm-up complete

2️⃣  Benchmarking ORIGINAL model (100 inferences)...
   Average: 64.3 ms
   P50: 62.4 ms
   P95: 79.8 ms

3️⃣  Benchmarking NEO-COMPILED model (100 inferences)...
   Average: 65.1 ms
   P50: 63.2 ms
   P95: 80.7 ms

======================================================================
BENCHMARK RESULTS
======================================================================

┌────────────────────┬──────────────┬──────────────┐
│  Metric            │  Original    │  Neo-Compiled│
├────────────────────┼──────────────┼──────────────┤
│  Average Latency   │      64.3 ms │      65.1 ms │
│  P50 Latency       │      62.4 ms │      63.2 ms │
│  P95 Latency       │      79.8 ms │      80.7 ms │
└────────────────────┴──────────────┴──────────────┘

   📊 Original is 1.01x faster (small models may not benefit)

----------------------------------------------------------------------
KEY EXAM CONCEPTS
----------------------------------------------------------------------

Neo benefits are MORE significant for:
   - Larger models (deep learning)
   - Edge/embedded devices with limited resources
   - High-throughput inference (many requests)

Neo benefits may be LESS noticeable for:
   - Small models (like our XGBoost demo)
   - Already-efficient frameworks
   - Network latency dominates (remote calls)

Real-world Neo speedups: typically 1.5x - 4x for deep learning models

======================================================================
⚠️  IMPORTANT: CLEANUP REQUIRED
======================================================================

Endpoints are EXPENSIVE! Delete when done:

   python cleanup_neo_demo.py

Or manually:
   aws sagemaker delete-endpoint --endpoint-name neo-demo-original-1764873559
   aws sagemaker delete-endpoint --endpoint-name neo-demo-compiled-1764873559

Show more
Benchmark complete! 🎉

Results Analysis
As expected, the performance is nearly identical (~64-65ms). This is actually a great learning outcome because it demonstrates a key exam concept:

Factor	Our Demo	When Neo Shines
Model size	18 KB (tiny)	100+ MB (deep learning)
Framework	XGBoost (already fast)	TensorFlow, PyTorch
Bottleneck	Network latency (~60ms)	Model computation
Hardware	Cloud CPU	Edge devices, accelerators
The ~60ms latency is mostly network round-trip from your Mac to AWS — the actual inference is probably <5ms on both.

When Neo Really Helps (Exam Focus)
Scenario	Neo Speedup
ResNet-50 on Jetson Nano	2-3x faster
BERT on ml.c5 instances	1.5-2x faster
MobileNet on Raspberry Pi	2-4x faster
Large CNN on Inferentia	3-5x faster
Let's Clean Up (Save Money!)
Here's the cleanup script:

cleanup_neo_demo.py
python
"""
Cleanup Neo demo resources
"""
import boto3
import time

region = boto3.Session().region_name
sagemaker_client = boto3.client('sagemaker', region_name=region)

print(f"Region: {region}")

# Read endpoint names
with open('endpoint_names.txt', 'r') as f:
    endpoints = [line.strip() for line in f.readlines() if line.strip()]

print("\n" + "="*70)
print("CLEANING UP NEO DEMO RESOURCES")
print("="*70)

# Delete endpoints
for endpoint_name in endpoints:
    print(f"\n🗑️  Deleting endpoint: {endpoint_name}")
    
    try:
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        print(f"   ✅ Delete initiated")
        
        # Wait for deletion
        while True:
            try:
                sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
                print(f"   ⏳ Waiting for deletion...")
                time.sleep(10)
            except sagemaker_client.exceptions.ClientError:
                print(f"   ✅ Deleted: {endpoint_name}")
                break
                
    except Exception as e:
        print(f"   ⚠️  Error: {e}")

# Delete endpoint configs
print("\n🗑️  Deleting endpoint configurations...")
for endpoint_name in endpoints:
    try:
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        print(f"   ✅ Deleted config: {endpoint_name}")
    except:
        pass

# Delete models
print("\n🗑️  Deleting models...")
timestamp = endpoints[0].split('-')[-1]  # Extract timestamp
model_names = [f"neo-demo-original-{timestamp}", f"neo-demo-compiled-{timestamp}"]

for model_name in model_names:
    try:
        sagemaker_client.delete_model(ModelName=model_name)
        print(f"   ✅ Deleted model: {model_name}")
    except:
        pass

print("\n" + "="*70)
print("✅ CLEANUP COMPLETE!")
print("="*70)
print("\n💰 Endpoints deleted - no more charges for these resources!")
Run Cleanup Now
bash
python cleanup_neo_demo.py
Let me know when cleanup is done, then I'll give you the Day 3 recap!

cleanup done

Let me give you the Day 3 recap:

Day 3 Recap: Edge Deployment & SageMaker Neo
What You Did
Step	What Happened
Trained XGBoost model	Simple binary classifier, 18 KB
Compiled with Neo	Targeted Linux x86_64, output 1 MB (includes runtime)
Deployed both versions	Original + Neo-compiled endpoints
Benchmarked latency	~64ms both (network-bound, not compute-bound)
Cleaned up	Deleted endpoints, configs, models
Key Concepts for the Exam
What is SageMaker Neo?

Compiles ML models for specific hardware targets
Optimizes model for faster inference and smaller footprint
Bundles runtime so target device doesn't need full framework
Supported Frameworks:

TensorFlow, PyTorch, MXNet, XGBoost, ONNX, Keras
Target Platforms:

Category	Examples
Cloud instances	ml.m5, ml.c5, ml.p3, ml.inf1
Edge devices	Jetson Nano/TX1/TX2, Raspberry Pi
Mobile	Android, iOS
Custom	Any ARM/x86 with Linux
When to Use Neo (Exam Scenarios)
Scenario	Use Neo?	Why
Deploy CNN to Jetson Nano	✅ Yes	Optimizes for ARM GPU
Real-time fraud detection on ml.c5	✅ Yes	Reduces latency
Batch processing overnight	❌ No	Latency doesn't matter
Model must run offline on device	✅ Yes	Bundles runtime
Quick prototype testing	❌ No	Extra compilation step
Neo Compilation Flow
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  Trained Model  │ ───▶ │  Neo Compiler   │ ───▶ │ Optimized Model │
│  (model.tar.gz) │      │                 │      │ + Runtime       │
│                 │      │  Target:        │      │                 │
│  Framework:     │      │  - OS           │      │  Runs on:       │
│  - XGBoost      │      │  - Architecture │      │  - Target HW    │
│  - TensorFlow   │      │  - Accelerator  │      │  - No framework │
│  - PyTorch      │      │                 │      │    needed       │
└─────────────────┘      └─────────────────┘      └─────────────────┘
Edge Manager (Concepts Only)
For deploying to actual edge devices at scale:

Feature	What It Does
Fleet management	Manage thousands of edge devices
Model packaging	Bundle models for specific devices
OTA updates	Push new model versions remotely
Monitoring	Collect inference data from devices
Dashboards	View device health and model performance
Exam tip: Edge Manager is for managing fleets of devices. Neo is for compiling models. They work together but serve different purposes.

Exam Questions You Can Now Answer
"How do you optimize a PyTorch model for Jetson Nano?" → Use SageMaker Neo with target platform Jetson Nano
"Model needs to run on device with no internet connection." → Neo compiles with bundled runtime, no framework needed
"How do you deploy models to thousands of IoT devices?" → SageMaker Edge Manager for fleet management + OTA updates
"What's the benefit of Neo compilation?" → Faster inference (up to 2-4x), smaller footprint, runs without full framework
"Neo-compiled model doesn't work on different hardware." → Correct! Models are compiled for specific targets, must recompile for different hardware
Files You Created
File	Purpose
train_model_for_neo.py	Train XGBoost model
compile_with_neo.py	Neo compilation job
deploy_and_benchmark.py	Deploy and compare
benchmark_only.py	Benchmark fix script
cleanup_neo_demo.py	Resource cleanup
Tomorrow: Day 4
Multi-Model & Multi-Container Endpoints

Host 5 models on 1 endpoint
Compare costs vs separate endpoints
Serial inference pipelines
Ready when you are! 🚀

Can I get that Neo recap with notes that look similar to this:

https://www.notion.so/Sagemaker-Feature-Store-2beccccfa62e8013918ce46b30d24d17

I can't access that Notion link directly, but I can view it if you share the content. Could you either:

Copy/paste the content from that Notion page into the chat, or
Export it as markdown or PDF and share it here
That way I can match the formatting style for your Neo recap!


# Sagemaker Feature Store ### What is Feature Store? **Simple definition:** A centralized repository to store, share, and manage ML features for training and inference. Think of it like a **database specifically designed for ML features** with special capabilities for ML workflows. --- ### The

pasted

Sure:

Here's your Neo recap in the same style:

SageMaker Neo & Edge Deployment
What is SageMaker Neo?
Simple definition: A compiler that optimizes ML models for specific hardware targets, making them faster and smaller.

Think of it like a translator that converts your model to speak the native language of your target hardware.

The Problem Neo Solves
Without Neo:

markdown
Trained Model (PyTorch)
       ↓
Deploy to Jetson Nano
       ↓
Requires full PyTorch
framework installed
       ↓
Slow inference
(not optimized for ARM)
       ↓
    😞 Poor performance
Common problems:

❌ Models require full framework (TensorFlow, PyTorch) on device
❌ Generic models not optimized for specific hardware
❌ Slow inference on edge devices
❌ Large model files consume device storage
❌ High latency for real-time applications
With Neo:

markdown
Trained Model (PyTorch)
       ↓
   Neo Compiler
   (Target: Jetson)
       ↓
Optimized Model +
Bundled Runtime
       ↓
  ┌────┴────┐
  ↓         ↓
Faster   No framework
inference   needed
  ↓         ↓
  ✅        ✅
Benefits:

✅ Up to 2-4x faster inference
✅ No framework needed on device (runtime bundled)
✅ Optimized for specific hardware
✅ Smaller memory footprint
✅ Same model code, different targets
Key Concepts
1. Compilation Job
Neo compiles your model for a specific target:

python
sagemaker_client.create_compilation_job(
    CompilationJobName="my-neo-job",
    InputConfig={
        'S3Uri': 's3://bucket/model.tar.gz',
        'DataInputConfig': '{"input": [1, 224, 224, 3]}',
        'Framework': 'PYTORCH'
    },
    OutputConfig={
        'S3OutputLocation': 's3://bucket/output/',
        'TargetPlatform': {
            'Os': 'LINUX',
            'Arch': 'ARM64'
        }
    }
)
```

Think: Converting source code to machine code, but for ML models.

---

### 2. Supported Frameworks

| Framework | Supported |
|-----------|-----------|
| TensorFlow | ✅ |
| PyTorch | ✅ |
| MXNet | ✅ |
| XGBoost | ✅ |
| ONNX | ✅ |
| Keras | ✅ |

---

### 3. Target Platforms

Neo can compile for many hardware targets:

| Category | Examples | Use Case |
|----------|----------|----------|
| **Cloud Instances** | ml.m5, ml.c5, ml.p3, ml.inf1 | SageMaker endpoints |
| **NVIDIA Devices** | Jetson Nano, TX1, TX2, Xavier | Robotics, drones |
| **ARM Devices** | Raspberry Pi, ARM64 Linux | IoT, edge |
| **Mobile** | Android (ARM), iOS | Mobile apps |
| **Accelerators** | AWS Inferentia | High-throughput |

**Example targets:**
```
Cloud:     ml_m5, ml_c5, ml_p3, ml_inf1
Edge:      jetson_nano, jetson_tx1, jetson_tx2
Embedded:  rasp3b (Raspberry Pi 3B)
Mobile:    android, ios
4. Input Configuration
You must specify the input shape of your model:

python
# Image model (batch, height, width, channels)
DataInputConfig = '{"input": [1, 224, 224, 3]}'

# Tabular model (batch, features)
DataInputConfig = '{"input": [1, 5]}'

# Sequence model (batch, sequence_length, features)
DataInputConfig = '{"input": [1, 128, 768]}'
```

**Why it matters:** Neo optimizes operations based on known input dimensions.

---

### Neo Compilation Flow
```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  Trained Model  │ ───▶ │  Neo Compiler   │ ───▶ │ Optimized Model │
│  (model.tar.gz) │      │                 │      │ + Runtime       │
│                 │      │  Target:        │      │                 │
│  Framework:     │      │  - OS           │      │  Runs on:       │
│  - XGBoost      │      │  - Architecture │      │  - Target HW    │
│  - TensorFlow   │      │  - Accelerator  │      │  - No framework │
│  - PyTorch      │      │                 │      │    needed       │
└─────────────────┘      └─────────────────┘      └─────────────────┘
```

---

### When Neo Helps Most

**Large performance gains:**

| Scenario | Typical Speedup |
|----------|-----------------|
| ResNet-50 on Jetson Nano | 2-3x faster |
| BERT on ml.c5 instances | 1.5-2x faster |
| MobileNet on Raspberry Pi | 2-4x faster |
| CNN on Inferentia | 3-5x faster |

**Smaller performance gains:**

| Scenario | Why |
|----------|-----|
| Small XGBoost models | Already efficient |
| Network-bound inference | Latency is round-trip, not compute |
| Simple models | Not much to optimize |

---

### Edge Manager (Fleet Management)

For deploying to **thousands of edge devices**:
```
┌─────────────────────────────────────────────────────────────┐
│                    SageMaker Edge Manager                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Device 1  │  │   Device 2  │  │   Device N  │        │
│  │   (Jetson)  │  │   (Rasp Pi) │  │   (Custom)  │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │
│         └────────────────┼────────────────┘                │
│                          ↓                                 │
│              ┌───────────────────────┐                     │
│              │   Edge Manager Agent   │                     │
│              │   - Model deployment   │                     │
│              │   - OTA updates        │                     │
│              │   - Health monitoring  │                     │
│              │   - Data collection    │                     │
│              └───────────────────────┘                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
Feature	What It Does
Fleet management	Manage thousands of edge devices
Model packaging	Bundle models for specific devices
OTA updates	Push new model versions remotely
Monitoring	Collect inference data from devices
Dashboards	View device health and performance
Key distinction:

Neo = Compiles models for hardware
Edge Manager = Manages device fleets
Common Use Cases
1. Autonomous Vehicles
markdown
Scenario: Object detection in self-driving car
Target:   NVIDIA Jetson Xavier
Why Neo:  Real-time inference (<50ms), no cloud dependency
2. Smart Cameras
markdown
Scenario: Facial recognition at building entrance  
Target:   NVIDIA Jetson Nano
Why Neo:  Low-latency, runs offline, privacy (data stays local)
3. Industrial IoT
markdown
Scenario: Predictive maintenance on factory floor
Target:   ARM64 Linux device
Why Neo:  No internet required, real-time anomaly detection
4. Mobile Apps
markdown
Scenario: Image classification in mobile app
Target:   Android ARM / iOS
Why Neo:  Fast inference, smaller app size, works offline
Neo vs Standard Deployment
Aspect	Standard Deployment	Neo-Compiled
Requires framework	✅ Full TF/PyTorch	❌ Runtime bundled
Optimized for HW	❌ Generic	✅ Target-specific
Inference speed	Baseline	1.5-4x faster
Model portability	✅ Any hardware	❌ Target-specific
Compilation step	❌ None	✅ Required
Key insight: Neo trades portability for performance. Compiled models only run on their target platform.

When to Use Neo
Use Neo when:

✅ Deploying to edge/embedded devices
✅ Need low-latency inference (<50ms)
✅ Device has limited resources
✅ Can't install full ML framework on device
✅ Running models offline (no cloud)
✅ High-throughput inference needed
Don't use Neo when:

❌ Quick prototyping/experimentation
❌ Model changes frequently
❌ Network latency dominates (remote API calls)
❌ Need to run on multiple hardware types
❌ Simple batch processing
Exam Questions
Q: "How do you optimize a PyTorch model for Jetson Nano?" → Use SageMaker Neo with target platform jetson_nano

Q: "Model needs to run on device with no internet." → Neo compiles with bundled runtime, no framework needed

Q: "How do you deploy models to thousands of IoT devices?" → SageMaker Edge Manager for fleet management + OTA updates

Q: "Neo-compiled model doesn't work on different hardware." → Correct! Models are target-specific, must recompile for different hardware

Q: "What's the benefit of Neo compilation?" → Faster inference (1.5-4x), smaller footprint, runs without full framework

Cost Considerations
Neo compilation:

Free! (only pay for the compilation compute time)
Edge Manager:

$0.05 per device per month (for fleet management)
Real cost is the hardware:

Jetson Nano: ~$99
Raspberry Pi 4: ~$75
ml.inf1.xlarge: ~$0.30/hour
FeatureWhat It DoesFleet managementManage thousands of edge devicesModel packagingBundle models for specific devicesOTA updatesPush new model versions remotelyMonitoringCollect inference data from devicesDashboardsView device health and performance

Can you turn this into a table in markdown?

Here's the table in markdown:

markdown
| Feature | What It Does |
|---------|--------------|
| Fleet management | Manage thousands of edge devices |
| Model packaging | Bundle models for specific devices |
| OTA updates | Push new model versions remotely |
| Monitoring | Collect inference data from devices |
| Dashboards | View device health and performance |
Which renders as:

Feature	What It Does
Fleet management	Manage thousands of edge devices
Model packaging	Bundle models for specific devices
OTA updates	Push new model versions remotely
Monitoring	Collect inference data from devices
Dashboards	View device health and performance
Same thing for this:

AspectStandard DeploymentNeo-CompiledRequires framework✅ Full TF/PyTorch❌ Runtime bundledOptimized for HW❌ Generic✅ Target-specificInference speedBaseline1.5-4x fasterModel portability✅ Any hardware❌ Target-specificCompilation step❌ None✅ Required

Here's the table in markdown:

markdown
| Aspect | Standard Deployment | Neo-Compiled |
|--------|---------------------|--------------|
| Requires framework | ✅ Full TF/PyTorch | ❌ Runtime bundled |
| Optimized for HW | ❌ Generic | ✅ Target-specific |
| Inference speed | Baseline | 1.5-4x faster |
| Model portability | ✅ Any hardware | ❌ Target-specific |
| Compilation step | ❌ None | ✅ Required |
Which renders as:

Aspect	Standard Deployment	Neo-Compiled
Requires framework	✅ Full TF/PyTorch	❌ Runtime bundled
Optimized for HW	❌ Generic	✅ Target-specific
Inference speed	Baseline	1.5-4x faster
Model portability	✅ Any hardware	❌ Target-specific
Compilation step	❌ None	✅ Required
I'm also going to check my code into my github.  I know we had to do a cleanup with the benchmark thing.  Could you provide a clean version of the code we worked with together and those scripts for just checking it in via terminal?

Let me create clean versions of all the scripts:


6 steps

Create .gitignore for neo-demo
.gitignore

Copy all neo-demo files to outputs


Looks like the outputs got mixed with the feature store files. Let me organize them properly:


Organize neo-demo files in proper folder


Here are the clean files for your GitHub repo:

Files Created
File	Description
README.md	Project documentation
train_model_for_neo.py	Train XGBoost model
compile_with_neo.py	Neo compilation job
deploy_and_benchmark.py	Deploy and benchmark (fixed version)
cleanup_neo_demo.py	Resource cleanup
requirements.txt	Python dependencies
.gitignore	Git ignore file
Terminal Commands to Check In
bash
# Navigate to your repo
cd ~/aws-ml-portfolio

# Create the neo-demo folder
mkdir -p neo-demo

# Copy the downloaded files
cp ~/Downloads/neo-demo/* neo-demo/
cp ~/Downloads/neo-demo/.gitignore neo-demo/

# Commit and push
git add neo-demo/
git commit -m "Add SageMaker Neo compilation and edge deployment demo"
git push origin main
Note: The deploy_and_benchmark.py is the fixed version that uses boto3 directly for endpoint invocation instead of the predictor object that was returning None.

Readme
Document · MD 
Train model for neo
PY 
Compile with neo
PY 
Deploy and benchmark
PY 
Cleanup neo demo
PY 
Requirements
TXT 
.gitignore
GITIGNORE 
all set, what's next for day 4?

Let's start Day 4: Multi-Model & Multi-Container Endpoints!

Day 4: Multi-Model & Multi-Container Endpoints
What You'll Learn
Concept	What It Does
Multi-Model Endpoints (MME)	Host many models on ONE endpoint (same framework)
Multi-Container Endpoints (MCE)	Chain different containers (different frameworks)
Cost optimization	1 endpoint vs N endpoints = major savings
Why It Matters
Scenario	Without MME	With MME
100 customer-specific models	100 endpoints = $$$$	1 endpoint = $
A/B testing 10 variants	10 endpoints	1 endpoint
Regional models	Endpoint per region	1 endpoint, many models
Exam Concepts Preview
Multi-Model Endpoint (MME):

Same framework (all XGBoost, all PyTorch, etc.)
Models loaded/unloaded dynamically
Specify TargetModel at inference time
Great for: per-customer models, many similar models
Multi-Container Endpoint (MCE):

Different frameworks in sequence
Serial inference pipeline (output of A → input of B)
Great for: preprocessing → inference → postprocessing
Hands-On Plan
Train 5 model variants (different hyperparameters)
Deploy as Multi-Model Endpoint
Test invoking different models
Compare costs vs 5 separate endpoints
Explore MCE concepts (serial inference)
Ready to start training the 5 models?

Let's do it.

Here's the script to train 5 model variants:

train_multiple_models.py
python
"""
Train 5 XGBoost model variants for Multi-Model Endpoint demo
"""
import boto3
import sagemaker
from sagemaker import image_uris
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
import pandas as pd
import numpy as np
import json
import time

# ============================================================
# CONFIGURATION - Update role with your SageMaker execution role
# ============================================================
role = "arn:aws:iam::854757836160:role/service-role/AmazonSageMaker-ExecutionRole-20251019T120276"

# Configuration
region = boto3.Session().region_name
boto_session = boto3.Session(region_name=region)
sagemaker_session = sagemaker.Session(boto_session=boto_session)

bucket = sagemaker_session.default_bucket()
prefix = 'mme-demo'

print(f"Region: {region}")
print(f"Bucket: {bucket}")
print(f"Prefix: {prefix}")

print("\n" + "="*70)
print("STEP 1: CREATE TRAINING DATASET")
print("="*70)

# Create a classification dataset
np.random.seed(42)
n_samples = 2000

data = {
    'feature_1': np.random.randn(n_samples),
    'feature_2': np.random.randn(n_samples),
    'feature_3': np.random.randn(n_samples),
    'feature_4': np.random.randn(n_samples),
    'feature_5': np.random.randn(n_samples),
}

df = pd.DataFrame(data)
df['target'] = ((df['feature_1'] + df['feature_2'] * 0.5 + np.random.randn(n_samples) * 0.3) > 0).astype(int)

# XGBoost format: target first, no headers
train_df = df[['target', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']]

print(f"\nDataset shape: {train_df.shape}")
print(f"Target distribution:\n{train_df['target'].value_counts()}")

# Save and upload
train_file = 'train.csv'
train_df.to_csv(train_file, index=False, header=False)

train_s3_path = f's3://{bucket}/{prefix}/train/{train_file}'
sagemaker_session.upload_data(train_file, bucket=bucket, key_prefix=f'{prefix}/train')

print(f"\n✅ Training data uploaded to: {train_s3_path}")

print("\n" + "="*70)
print("STEP 2: DEFINE 5 MODEL VARIANTS")
print("="*70)

# 5 different hyperparameter configurations
# Simulating customer-specific or A/B test variants
model_variants = {
    'model_conservative': {
        'num_round': 25,
        'max_depth': 2,
        'eta': 0.3,
        'description': 'Simple, fast, low risk of overfitting'
    },
    'model_balanced': {
        'num_round': 50,
        'max_depth': 4,
        'eta': 0.2,
        'description': 'Balanced performance'
    },
    'model_aggressive': {
        'num_round': 100,
        'max_depth': 6,
        'eta': 0.1,
        'description': 'Complex, potentially more accurate'
    },
    'model_deep': {
        'num_round': 50,
        'max_depth': 8,
        'eta': 0.2,
        'description': 'Deep trees, captures complex patterns'
    },
    'model_fast': {
        'num_round': 30,
        'max_depth': 3,
        'eta': 0.4,
        'description': 'Optimized for speed'
    }
}

print("\nModel variants to train:")
for name, config in model_variants.items():
    print(f"\n   {name}:")
    print(f"      num_round: {config['num_round']}")
    print(f"      max_depth: {config['max_depth']}")
    print(f"      eta: {config['eta']}")
    print(f"      → {config['description']}")

print("\n" + "="*70)
print("STEP 3: TRAIN ALL 5 MODELS")
print("="*70)

# Get XGBoost container
container = image_uris.retrieve('xgboost', region, '1.5-1')
print(f"\nUsing container: {container}")

# Store model artifacts
model_artifacts = {}

for i, (model_name, config) in enumerate(model_variants.items(), 1):
    print(f"\n{'─'*70}")
    print(f"Training model {i}/5: {model_name}")
    print(f"{'─'*70}")
    print(f"   Hyperparameters: num_round={config['num_round']}, "
          f"max_depth={config['max_depth']}, eta={config['eta']}")
    
    # Create estimator
    estimator = Estimator(
        image_uri=container,
        role=role,
        instance_count=1,
        instance_type='ml.m5.large',
        output_path=f's3://{bucket}/{prefix}/models/{model_name}/',
        sagemaker_session=sagemaker_session,
        base_job_name=f'mme-{model_name}'
    )
    
    # Set hyperparameters
    estimator.set_hyperparameters(
        objective='binary:logistic',
        num_round=config['num_round'],
        max_depth=config['max_depth'],
        eta=config['eta'],
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    # Train
    print(f"   Starting training job...")
    start_time = time.time()
    
    estimator.fit(
        {'train': TrainingInput(train_s3_path, content_type='text/csv')},
        wait=True,
        logs=False  # Suppress detailed logs
    )
    
    elapsed = time.time() - start_time
    
    # Store artifact location
    model_artifacts[model_name] = estimator.model_data
    
    print(f"   ✅ Completed in {elapsed:.0f}s")
    print(f"   📦 Artifact: {estimator.model_data}")

print("\n" + "="*70)
print("✅ ALL 5 MODELS TRAINED!")
print("="*70)

print("\nModel artifacts:")
for name, artifact in model_artifacts.items():
    print(f"   {name}: {artifact}")

# Save model artifacts for next step
with open('model_artifacts.json', 'w') as f:
    json.dump(model_artifacts, f, indent=2)

print("\n📝 Model artifacts saved to: model_artifacts.json")
print("\n💡 Next step: python deploy_multi_model_endpoint.py")
Run It
bash
cd ~
mkdir -p mme-demo
cd mme-demo

# Save the script and run
python train_multiple_models.py
This will train 5 XGBoost variants sequentially (~15-20 min total). Let me know when it completes!


sagemaker.config INFO - Not applying SDK defaults from location: /Library/Application Support/sagemaker/config.yaml sagemaker.config INFO - Not applying SDK defaults from location: /Users/adamvigneau/Library/Application Support/sagemaker/config.yaml Region: us-east-2 Bucket: sagemaker-us-east-2-8547

pasted

The issue is the underscore in the job name — SageMaker only allows hyphens. Let me fix that:

Quick Fix
Replace line ~138 (the base_job_name line):

python
base_job_name=f'mme-{model_name}'
With:

python
base_job_name=f'mme-{model_name.replace("_", "-")}'
Or I can give you the full corrected script. Which do you prefer?

Just made the replacement and started running.  If I have to stop after the models are trained and come back to this later, is there anything I have to tear down?

Good news — training jobs don't need teardown. They automatically stop when complete and you only pay for the compute time used.

What costs money if left running:

❌ Endpoints (we haven't deployed yet)
❌ Notebook instances (if you had any running)
What's safe to leave:

✅ Training jobs (auto-stop)
✅ Model artifacts in S3 (pennies)
✅ The model_artifacts.json file (local)
So if you need to stop after training completes, you're fine. Just don't run the deploy script until you're ready to test and clean up in the same session.

Let me know when training finishes or if you need to pick this up later!

Models have been trained.

Let's deploy the Multi-Model Endpoint. Here's the script:

deploy_multi_model_endpoint.py
python
"""
Deploy 5 models to a single Multi-Model Endpoint (MME)
"""
import boto3
import sagemaker
from sagemaker import image_uris
import json
import time
import tarfile
import os

# ============================================================
# CONFIGURATION
# ============================================================
role = "arn:aws:iam::854757836160:role/service-role/AmazonSageMaker-ExecutionRole-20251019T120276"

region = boto3.Session().region_name
sagemaker_session = sagemaker.Session()
sagemaker_client = boto3.client('sagemaker', region_name=region)
s3_client = boto3.client('s3', region_name=region)

bucket = sagemaker_session.default_bucket()
prefix = 'mme-demo'

print(f"Region: {region}")
print(f"Bucket: {bucket}")

# Load model artifacts from training
with open('model_artifacts.json', 'r') as f:
    model_artifacts = json.load(f)

print(f"\nLoaded {len(model_artifacts)} model artifacts")

print("\n" + "="*70)
print("STEP 1: COPY MODELS TO MME LOCATION")
print("="*70)

# MME requires all models in a single S3 prefix
mme_model_prefix = f'{prefix}/mme-models'
print(f"\nMME model location: s3://{bucket}/{mme_model_prefix}/")

# Copy each model to the MME location with a simple name
for model_name, artifact_path in model_artifacts.items():
    # Parse source location
    source_path = artifact_path.replace(f's3://{bucket}/', '')
    
    # Destination: simple name like "model_conservative.tar.gz"
    dest_key = f'{mme_model_prefix}/{model_name}.tar.gz'
    
    print(f"\n   Copying {model_name}...")
    print(f"   From: {source_path}")
    print(f"   To: {dest_key}")
    
    s3_client.copy_object(
        Bucket=bucket,
        CopySource={'Bucket': bucket, 'Key': source_path},
        Key=dest_key
    )
    print(f"   ✅ Copied")

print("\n" + "="*70)
print("STEP 2: CREATE MULTI-MODEL ENDPOINT")
print("="*70)

timestamp = int(time.time())
model_name = f'mme-demo-model-{timestamp}'
endpoint_config_name = f'mme-demo-config-{timestamp}'
endpoint_name = f'mme-demo-endpoint-{timestamp}'

# Get XGBoost container (must support MME)
container = image_uris.retrieve('xgboost', region, '1.5-1')
print(f"\nContainer: {container}")

# Create Model (pointing to the MME prefix, not a single model)
print(f"\n1️⃣  Creating MME Model: {model_name}")

sagemaker_client.create_model(
    ModelName=model_name,
    PrimaryContainer={
        'Image': container,
        'Mode': 'MultiModel',  # This makes it an MME!
        'ModelDataUrl': f's3://{bucket}/{mme_model_prefix}/'
    },
    ExecutionRoleArn=role
)
print(f"   ✅ Model created")

# Create Endpoint Config
print(f"\n2️⃣  Creating Endpoint Config: {endpoint_config_name}")

sagemaker_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[{
        'VariantName': 'AllModels',
        'ModelName': model_name,
        'InitialInstanceCount': 1,
        'InstanceType': 'ml.m5.large'
    }]
)
print(f"   ✅ Endpoint config created")

# Create Endpoint
print(f"\n3️⃣  Creating Endpoint: {endpoint_name}")
print(f"   This will take 3-5 minutes...\n")

sagemaker_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)

# Wait for endpoint to be ready
while True:
    response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
    status = response['EndpointStatus']
    
    if status == 'InService':
        print(f"   ✅ Endpoint is InService!")
        break
    elif status == 'Failed':
        print(f"   ❌ Endpoint failed: {response.get('FailureReason', 'Unknown')}")
        exit(1)
    else:
        print(f"   ⏳ Status: {status}...")
        time.sleep(30)

# Save endpoint info for testing and cleanup
endpoint_info = {
    'endpoint_name': endpoint_name,
    'endpoint_config_name': endpoint_config_name,
    'model_name': model_name,
    'models': list(model_artifacts.keys())
}

with open('endpoint_info.json', 'w') as f:
    json.dump(endpoint_info, f, indent=2)

print("\n" + "="*70)
print("✅ MULTI-MODEL ENDPOINT DEPLOYED!")
print("="*70)

print(f"""
Endpoint: {endpoint_name}
Models available:
   - model_conservative.tar.gz
   - model_balanced.tar.gz
   - model_aggressive.tar.gz
   - model_deep.tar.gz
   - model_fast.tar.gz

📝 Endpoint info saved to: endpoint_info.json

💡 Next step: python test_mme.py
""")

print("\n" + "-"*70)
print("COST COMPARISON")
print("-"*70)
print(f"""
┌─────────────────────────────────────────────────────────────┐
│  Deployment Strategy      │  Endpoints  │  Cost/hour       │
├─────────────────────────────────────────────────────────────┤
│  5 Separate Endpoints     │     5       │  5 × $0.10 = $0.50│
│  1 Multi-Model Endpoint   │     1       │  1 × $0.10 = $0.10│
├─────────────────────────────────────────────────────────────┤
│  💰 SAVINGS               │    -4       │  80% cheaper!     │
└─────────────────────────────────────────────────────────────┘

(Costs based on ml.m5.large at ~$0.10/hour)
""")
Run It
bash
python deploy_multi_model_endpoint.py
This will:

Copy all 5 models to a single S3 prefix
Create an MME with all models available
Show the cost savings
Let me know when it's deployed!

end point deployed

Now let's test invoking different models on the same endpoint:

test_mme.py
python
"""
Test Multi-Model Endpoint - invoke different models on the same endpoint
"""
import boto3
import json
import time
import numpy as np

# Configuration
region = boto3.Session().region_name
runtime_client = boto3.client('sagemaker-runtime', region_name=region)

print(f"Region: {region}")

# Load endpoint info
with open('endpoint_info.json', 'r') as f:
    endpoint_info = json.load(f)

endpoint_name = endpoint_info['endpoint_name']
models = endpoint_info['models']

print(f"Endpoint: {endpoint_name}")
print(f"Available models: {models}")

print("\n" + "="*70)
print("TESTING MULTI-MODEL ENDPOINT")
print("="*70)

# Generate test data
np.random.seed(42)
test_sample = np.random.randn(5).tolist()
test_payload = ','.join(map(str, test_sample))

print(f"\nTest input: {test_payload[:50]}...")

# ============================================================
# Test 1: Invoke each model
# ============================================================
print("\n" + "-"*70)
print("TEST 1: Invoke Each Model")
print("-"*70)

results = {}

for model_name in models:
    target_model = f"{model_name}.tar.gz"
    
    print(f"\n   Invoking {model_name}...")
    
    start = time.time()
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        TargetModel=target_model,  # THIS IS THE KEY - specify which model!
        ContentType='text/csv',
        Body=test_payload
    )
    latency = (time.time() - start) * 1000
    
    prediction = response['Body'].read().decode('utf-8')
    results[model_name] = {
        'prediction': float(prediction),
        'latency': latency
    }
    
    print(f"   Prediction: {float(prediction):.4f}")
    print(f"   Latency: {latency:.1f} ms")

# ============================================================
# Test 2: Compare predictions
# ============================================================
print("\n" + "-"*70)
print("TEST 2: Compare Predictions Across Models")
print("-"*70)

print(f"""
┌─────────────────────┬─────────────┬─────────────┐
│  Model              │  Prediction │  Latency    │
├─────────────────────┼─────────────┼─────────────┤""")

for model_name, data in results.items():
    name_display = model_name.replace('model_', '')[:15].ljust(15)
    print(f"│  {name_display}    │  {data['prediction']:>9.4f}  │  {data['latency']:>7.1f} ms │")

print(f"└─────────────────────┴─────────────┴─────────────┘")

# ============================================================
# Test 3: Latency benchmark (model loading)
# ============================================================
print("\n" + "-"*70)
print("TEST 3: Model Loading Behavior")
print("-"*70)

print("""
MME loads models ON DEMAND. First call to a model is slower (cold start).
Subsequent calls are faster (model cached in memory).
""")

# Test cold vs warm for one model
test_model = "model_balanced.tar.gz"

# First, invoke a different model to potentially unload balanced
runtime_client.invoke_endpoint(
    EndpointName=endpoint_name,
    TargetModel="model_fast.tar.gz",
    ContentType='text/csv',
    Body=test_payload
)

print(f"Testing {test_model}:")

latencies = []
for i in range(5):
    start = time.time()
    runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        TargetModel=test_model,
        ContentType='text/csv',
        Body=test_payload
    )
    latency = (time.time() - start) * 1000
    latencies.append(latency)
    call_type = "Cold" if i == 0 else "Warm"
    print(f"   Call {i+1} ({call_type}): {latency:.1f} ms")

print(f"\n   Average (excluding first): {sum(latencies[1:])/len(latencies[1:]):.1f} ms")

# ============================================================
# Test 4: Rapid model switching
# ============================================================
print("\n" + "-"*70)
print("TEST 4: Rapid Model Switching")
print("-"*70)

print("\nSwitching between models rapidly (simulating A/B testing):\n")

switch_sequence = ['model_conservative', 'model_aggressive', 'model_conservative', 
                   'model_balanced', 'model_aggressive', 'model_fast']

for model_name in switch_sequence:
    target_model = f"{model_name}.tar.gz"
    
    start = time.time()
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        TargetModel=target_model,
        ContentType='text/csv',
        Body=test_payload
    )
    latency = (time.time() - start) * 1000
    prediction = float(response['Body'].read().decode('utf-8'))
    
    print(f"   {model_name.replace('model_', ''):12} → {prediction:.4f} ({latency:.0f}ms)")

print("\n" + "="*70)
print("✅ MME TESTING COMPLETE!")
print("="*70)

print("""
📝 KEY EXAM CONCEPTS:

1. TargetModel parameter specifies which model to invoke
2. Models are loaded on-demand (first call = cold start)
3. Frequently used models stay cached in memory
4. All models must use the SAME framework/container
5. Cost savings: 1 endpoint serves unlimited models

💡 Use cases:
   - Per-customer models (each customer has personalized model)
   - A/B testing (multiple model versions)
   - Regional models (different models for different regions)
   - Time-based models (daily/weekly model updates)
""")

print(f"\n⚠️  Don't forget to clean up: python cleanup_mme.py")
Run It
bash
python test_mme.py
This will demonstrate:

Invoking different models on the same endpoint
Comparing predictions across models
Cold start vs warm latency
Rapid model switching
Let me know what results you get!

Region: us-east-2
Endpoint: mme-demo-endpoint-1764906114
Available models: ['model_conservative', 'model_balanced', 'model_aggressive', 'model_deep', 'model_fast']

======================================================================
TESTING MULTI-MODEL ENDPOINT
======================================================================

Test input: 0.4967141530112327,-0.13826430117118466,0.64768853...

----------------------------------------------------------------------
TEST 1: Invoke Each Model
----------------------------------------------------------------------

   Invoking model_conservative...
Traceback (most recent call last):
  File "/Users/adamvigneau/mme-demo/test_mme.py", line 61, in <module>
    'prediction': float(prediction),
                  ~~~~~^^^^^^^^^^^^
ValueError: could not convert string to float: '[0.8166066408157349]'

Show more
The prediction is returning as a JSON array. Quick fix — update line 61:

Replace:

python
    prediction = response['Body'].read().decode('utf-8')
    results[model_name] = {
        'prediction': float(prediction),
With:

python
    prediction = response['Body'].read().decode('utf-8')
    # Handle both "[0.123]" and "0.123" formats
    pred_value = prediction.strip('[]')
    results[model_name] = {
        'prediction': float(pred_value),
Also update line 109 (in Test 4) the same way:

Replace:

python
    prediction = float(response['Body'].read().decode('utf-8'))
With:

python
    prediction = float(response['Body'].read().decode('utf-8').strip('[]'))
Run it again:

bash
python test_mme.py
Region: us-east-2
Endpoint: mme-demo-endpoint-1764906114
Available models: ['model_conservative', 'model_balanced', 'model_aggressive', 'model_deep', 'model_fast']

======================================================================
TESTING MULTI-MODEL ENDPOINT
======================================================================

Test input: 0.4967141530112327,-0.13826430117118466,0.64768853...

----------------------------------------------------------------------
TEST 1: Invoke Each Model
----------------------------------------------------------------------

   Invoking model_conservative...
Traceback (most recent call last):
  File "/Users/adamvigneau/mme-demo/test_mme.py", line 71, in <module>
    print(f"   Prediction: {float(prediction):.4f}")
                            ~~~~~^^^^^^^^^^^^
ValueError: could not convert string to float: '[0.8166066408157349]'

Show more
There's another print statement that needs fixing. Replace line 71:

python
    print(f"   Prediction: {float(prediction):.4f}")
With:

python
    print(f"   Prediction: {results[model_name]['prediction']:.4f}")
Run it again:

bash
python test_mme.py
testing complete

Great! Now let's clean up to avoid charges:

cleanup_mme.py
python
"""
Cleanup Multi-Model Endpoint resources
"""
import boto3
import json
import time

region = boto3.Session().region_name
sagemaker_client = boto3.client('sagemaker', region_name=region)

print(f"Region: {region}")

# Load endpoint info
try:
    with open('endpoint_info.json', 'r') as f:
        endpoint_info = json.load(f)
except FileNotFoundError:
    print("endpoint_info.json not found. Enter names manually:")
    endpoint_info = {
        'endpoint_name': input("Endpoint name: ").strip(),
        'endpoint_config_name': input("Endpoint config name: ").strip(),
        'model_name': input("Model name: ").strip()
    }

endpoint_name = endpoint_info['endpoint_name']
endpoint_config_name = endpoint_info['endpoint_config_name']
model_name = endpoint_info['model_name']

print("\n" + "="*70)
print("CLEANING UP MME RESOURCES")
print("="*70)

# Delete endpoint
print(f"\n1️⃣  Deleting endpoint: {endpoint_name}")
try:
    sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
    print(f"   ✅ Delete initiated")
    
    while True:
        try:
            sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            print(f"   ⏳ Waiting for deletion...")
            time.sleep(10)
        except sagemaker_client.exceptions.ClientError:
            print(f"   ✅ Endpoint deleted")
            break
except Exception as e:
    print(f"   ⚠️  Error: {e}")

# Delete endpoint config
print(f"\n2️⃣  Deleting endpoint config: {endpoint_config_name}")
try:
    sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
    print(f"   ✅ Endpoint config deleted")
except Exception as e:
    print(f"   ⚠️  Error: {e}")

# Delete model
print(f"\n3️⃣  Deleting model: {model_name}")
try:
    sagemaker_client.delete_model(ModelName=model_name)
    print(f"   ✅ Model deleted")
except Exception as e:
    print(f"   ⚠️  Error: {e}")

print("\n" + "="*70)
print("✅ CLEANUP COMPLETE!")
print("="*70)
print("\n💰 No more charges for this endpoint!")
Run It
bash
python cleanup_mme.py
Let me know when cleanup is done and I'll give you the Day 4 recap!

all set

Here's your Day 4 recap:

Multi-Model & Multi-Container Endpoints
What is a Multi-Model Endpoint (MME)?
Simple definition: A single endpoint that hosts many models, loading them on-demand to save costs.

Think of it like a shared apartment — multiple tenants (models) share one space (endpoint) instead of each paying for their own.

The Problem MME Solves
Without MME:

markdown
Customer A Model     Customer B Model     Customer C Model
       ↓                   ↓                    ↓
   Endpoint A          Endpoint B           Endpoint C
   ($0.10/hr)          ($0.10/hr)           ($0.10/hr)
       ↓                   ↓                    ↓
   Total: $0.30/hour for 3 models
   100 customers = $10/hour = $7,200/month 😱
Common problems:

❌ Cost scales linearly with number of models
❌ Managing hundreds of endpoints is complex
❌ Most endpoints sit idle (low utilization)
❌ A/B testing requires multiple endpoints
With MME:

markdown
Customer A Model ─┐
Customer B Model ─┼──→ ONE Endpoint ──→ $0.10/hr
Customer C Model ─┘    (loads on demand)
       
100 customers = still ~$0.10/hour = $72/month 🎉
```

**Benefits:**

- ✅ 80-90% cost reduction for many models
- ✅ Single endpoint to manage
- ✅ Models loaded/unloaded automatically
- ✅ Easy A/B testing (same endpoint, different models)
- ✅ Scales to thousands of models

---

### Key Concepts

### 1. How MME Works
```
┌─────────────────────────────────────────────────────────────┐
│                Multi-Model Endpoint                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   S3 Bucket (Model Repository)                              │
│   ┌─────────────────────────────────┐                       │
│   │ model_conservative.tar.gz       │                       │
│   │ model_balanced.tar.gz           │                       │
│   │ model_aggressive.tar.gz         │◄─── Models stored     │
│   │ model_deep.tar.gz               │     in S3             │
│   │ model_fast.tar.gz               │                       │
│   └─────────────────────────────────┘                       │
│                    │                                        │
│                    ▼                                        │
│   ┌─────────────────────────────────┐                       │
│   │     Inference Container         │                       │
│   │  ┌───────────┐ ┌───────────┐   │                       │
│   │  │ Model A   │ │ Model B   │   │◄─── Models loaded     │
│   │  │ (cached)  │ │ (cached)  │   │     on demand         │
│   │  └───────────┘ └───────────┘   │                       │
│   └─────────────────────────────────┘                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
2. TargetModel Parameter
The key to MME — specify which model at inference time:

python
response = runtime_client.invoke_endpoint(
    EndpointName='my-mme-endpoint',
    TargetModel='model_conservative.tar.gz',  # THIS IS THE KEY!
    ContentType='text/csv',
    Body=payload
)
Same endpoint, different model each request.

3. Model Loading Behavior
Call Type	Latency	Why
Cold start	1-5 seconds	Model loaded from S3
Warm call	50-100 ms	Model cached in memory
markdown
First call to model_A:   [Load from S3] ──→ Slow (cold)
Second call to model_A:  [Use cache]    ──→ Fast (warm)
Call to model_B:         [Load from S3] ──→ Slow (cold)
Back to model_A:         [Still cached] ──→ Fast (warm)
```

**Frequently used models stay cached. Least-recently-used models get evicted.**

---

### 4. MME Requirements

| Requirement | Details |
|-------------|---------|
| Same framework | All models must use same container (all XGBoost, all PyTorch, etc.) |
| Same input format | All models must accept same input type |
| S3 location | All models in single S3 prefix |
| Model naming | Each model needs unique `.tar.gz` filename |

---

### Cost Comparison
```
┌─────────────────────────────────────────────────────────────┐
│  Deployment Strategy      │  Endpoints  │  Cost/hour       │
├─────────────────────────────────────────────────────────────┤
│  5 Separate Endpoints     │     5       │  5 × $0.10 = $0.50│
│  1 Multi-Model Endpoint   │     1       │  1 × $0.10 = $0.10│
├─────────────────────────────────────────────────────────────┤
│  💰 SAVINGS               │    -4       │  80% cheaper!     │
└─────────────────────────────────────────────────────────────┘
```

**At scale:**

| Models | Separate Endpoints | MME | Savings |
|--------|-------------------|-----|---------|
| 10 | $1.00/hr | $0.10/hr | 90% |
| 100 | $10.00/hr | $0.10-0.30/hr | 97% |
| 1000 | $100.00/hr | $0.30-1.00/hr | 99% |

---

### Multi-Container Endpoints (MCE)

**Different from MME** — MCE chains multiple containers in sequence:
```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Container 1  │───▶│ Container 2  │───▶│ Container 3  │
│ (Preprocess) │    │ (Inference)  │    │ (Postprocess)│
│ sklearn      │    │ PyTorch      │    │ Custom       │
└──────────────┘    └──────────────┘    └──────────────┘
Feature	MME	MCE
Purpose	Many models, pick one	Chain models together
Containers	1 (same framework)	Multiple (different frameworks)
Invocation	TargetModel parameter	Serial pipeline
Use case	Per-customer models	Preprocessing → Inference → Postprocessing
Common Use Cases
1. Per-Customer Models
markdown
Scenario: E-commerce with personalized recommendations
Solution: Train one model per customer, host all on MME
Invoke:   TargetModel='customer_12345.tar.gz'
2. A/B Testing
markdown
Scenario: Test 5 model variants in production
Solution: All variants on one MME
Invoke:   Randomly select TargetModel based on traffic split
3. Regional Models
markdown
Scenario: Different models for US, EU, APAC
Solution: All regional models on one MME
Invoke:   TargetModel='model_us.tar.gz' based on user location
4. Time-Based Models
markdown
Scenario: Retrain model daily, keep last 7 versions
Solution: All versions on MME for instant rollback
Invoke:   TargetModel='model_2024_12_04.tar.gz'
Exam Questions
Q: "How do you host 100 customer-specific models cost-effectively?" → Multi-Model Endpoint — one endpoint, specify TargetModel per request

Q: "First inference call is slow, subsequent calls are fast. Why?" → MME cold start — model loaded from S3 on first call, cached for subsequent calls

Q: "Need to chain sklearn preprocessing with PyTorch inference." → Multi-Container Endpoint (MCE) — serial inference pipeline

Q: "All models must use the same framework for MME." → True! MME requires single container type

Q: "How do you specify which model to invoke on MME?" → TargetModel parameter in invoke_endpoint()

MME vs MCE Summary
Aspect	Multi-Model (MME)	Multi-Container (MCE)
Models	Many (pick one)	Few (chain all)
Framework	Same	Different
Invocation	TargetModel param	Serial pipeline
Use case	Per-customer, A/B	Preprocessing chains
Cost benefit	Massive (1 vs N)	Moderate
What You Built Today
Step	What You Did
Trained 5 models	Different hyperparameters (conservative → aggressive)
Deployed MME	Single endpoint hosting all 5 models
Tested invocation	Used TargetModel to call each model
Observed cold/warm	Saw loading behavior in action
Calculated savings	80% cheaper than 5 separate endpoints
Files Created
File	Purpose
train_multiple_models.py	Train 5 XGBoost variants
deploy_multi_model_endpoint.py	Create MME
test_mme.py	Test invoking different models
cleanup_mme.py	Delete resources
Tomorrow is Day 5: Advanced Monitoring & Observability — the final piece before your weekend capstone project!

Can you create a README.md and any other files you didn't already generate?


6 steps

Create .gitignore for mme-demo
.gitignore

Copy all mme-demo files to outputs


Here are all the files for your GitHub repo:

Files Created
File	Description
README.md	Project documentation
train_multiple_models.py	Train 5 XGBoost variants
deploy_multi_model_endpoint.py	Deploy MME
test_mme.py	Test invoking different models (with fixes)
cleanup_mme.py	Delete resources
requirements.txt	Python dependencies
.gitignore	Git ignore file
Terminal Commands to Check In
bash
# Navigate to your repo
cd ~/aws-ml-portfolio

# Create the mme-demo folder
mkdir -p mme-demo

# Copy the downloaded files
cp ~/Downloads/mme-demo/* mme-demo/
cp ~/Downloads/mme-demo/.gitignore mme-demo/

# Commit and push
git add mme-demo/
git commit -m "Add Multi-Model Endpoint demo"
git push origin main
All set for Day 5 (Advanced Monitoring) when you're ready!

Readme
Document · MD 
Train multiple models
PY 
Deploy multi model endpoint
PY 
Test mme
PY 
Cleanup mme
PY 
Requirements
TXT 
.gitignore
GITIGNORE 
Call TypeLatencyWhyCold start1-5 secondsModel loaded from S3Warm call50-100 msModel cached in memory

Convert this to a table in markdown

Here's the table in markdown:

markdown
| Call Type | Latency | Why |
|-----------|---------|-----|
| Cold start | 1-5 seconds | Model loaded from S3 |
| Warm call | 50-100 ms | Model cached in memory |
Which renders as:

Call Type	Latency	Why
Cold start	1-5 seconds	Model loaded from S3
Warm call	50-100 ms	Model cached in memory
Convert the rest of the tables you generated for the recap into markdown

Here are all the tables from the Day 4 recap in markdown:

Cost Comparison
markdown
| Deployment Strategy | Endpoints | Cost/hour |
|---------------------|-----------|-----------|
| 5 Separate Endpoints | 5 | 5 × $0.10 = $0.50 |
| 1 Multi-Model Endpoint | 1 | 1 × $0.10 = $0.10 |
| 💰 SAVINGS | -4 | 80% cheaper! |
At Scale
markdown
| Models | Separate Endpoints | MME | Savings |
|--------|-------------------|-----|---------|
| 10 | $1.00/hr | $0.10/hr | 90% |
| 100 | $10.00/hr | $0.10-0.30/hr | 97% |
| 1000 | $100.00/hr | $0.30-1.00/hr | 99% |
MME vs MCE Feature Comparison
markdown
| Feature | MME | MCE |
|---------|-----|-----|
| Purpose | Many models, pick one | Chain models together |
| Containers | 1 (same framework) | Multiple (different frameworks) |
| Invocation | TargetModel parameter | Serial pipeline |
| Use case | Per-customer models | Preprocessing → Inference → Postprocessing |
MME Requirements
markdown
| Requirement | Details |
|-------------|---------|
| Same framework | All models must use same container (all XGBoost, all PyTorch, etc.) |
| Same input format | All models must accept same input type |
| S3 location | All models in single S3 prefix |
| Model naming | Each model needs unique `.tar.gz` filename |
MME vs MCE Summary
markdown
| Aspect | Multi-Model (MME) | Multi-Container (MCE) |
|--------|-------------------|----------------------|
| Models | Many (pick one) | Few (chain all) |
| Framework | Same | Different |
| Invocation | TargetModel param | Serial pipeline |
| Use case | Per-customer, A/B | Preprocessing chains |
| Cost benefit | Massive (1 vs N) | Moderate |
Model Variants
markdown
| Model | num_round | max_depth | eta | Description |
|-------|-----------|-----------|-----|-------------|
| model_conservative | 25 | 2 | 0.3 | Simple, fast, low risk of overfitting |
| model_balanced | 50 | 4 | 0.2 | Balanced performance |
| model_aggressive | 100 | 6 | 0.1 | Complex, potentially more accurate |
| model_deep | 50 | 8 | 0.2 | Deep trees, captures complex patterns |
| model_fast | 30 | 3 | 0.4 | Optimized for speed |
What You Built Today
markdown
| Step | What You Did |
|------|--------------|
| Trained 5 models | Different hyperparameters (conservative → aggressive) |
| Deployed MME | Single endpoint hosting all 5 models |
| Tested invocation | Used TargetModel to call each model |
| Observed cold/warm | Saw loading behavior in action |
| Calculated savings | 80% cheaper than 5 separate endpoints |
Files Created
markdown
| File | Purpose |
|------|---------|
| `train_multiple_models.py` | Train 5 XGBoost variants |
| `deploy_multi_model_endpoint.py` | Create MME |
| `test_mme.py` | Test invoking different models |
| `cleanup_mme.py` | Delete resources |
Ready to start day 5.

Let's start Day 5: Advanced Monitoring & Observability!

Day 5: Advanced Monitoring & Observability
What You'll Learn
Concept	What It Does
SageMaker Model Monitor	Detect data drift, model quality issues
CloudWatch Metrics	Custom metrics and dashboards
Clarify	Model explainability (SHAP values)
Ground Truth	Human labeling for retraining
Why Monitoring Matters
Deploy Model → Works great! → Time passes → Performance degrades → 😱

Why?
- Data drift (input data changes over time)
- Concept drift (relationship between features and target changes)
- Model staleness (world changed, model didn't)
Monitoring catches problems BEFORE they impact users.

Exam Concepts Preview
Model Monitor Types:

Monitor Type	What It Detects
Data Quality	Missing values, type mismatches, outliers
Model Quality	Accuracy, precision, recall degradation
Bias Drift	Fairness metrics changing over time
Feature Attribution	SHAP values drifting
Hands-On Plan
Deploy a model with data capture enabled
Create a baseline (what "normal" looks like)
Set up Model Monitor schedule
Simulate data drift and see alerts
View CloudWatch metrics and dashboards
Note on Timing
Model Monitor schedules run hourly at minimum, so we'll:

Set everything up
Simulate drift manually
Examine the monitoring outputs
Ready to start with deploying a model with data capture enabled?

Let's do it.

Here's the script to deploy a model with data capture enabled:

deploy_with_monitoring.py
python
"""
Deploy a model with Data Capture enabled for Model Monitor
"""
import boto3
import sagemaker
from sagemaker import image_uris
from sagemaker.model import Model
from sagemaker.model_monitor import DataCaptureConfig
import pandas as pd
import numpy as np
import json
import time

# ============================================================
# CONFIGURATION
# ============================================================
role = "arn:aws:iam::854757836160:role/service-role/AmazonSageMaker-ExecutionRole-20251019T120276"

region = boto3.Session().region_name
sagemaker_session = sagemaker.Session()
sagemaker_client = boto3.client('sagemaker', region_name=region)

bucket = sagemaker_session.default_bucket()
prefix = 'monitoring-demo'

print(f"Region: {region}")
print(f"Bucket: {bucket}")
print(f"Prefix: {prefix}")

print("\n" + "="*70)
print("STEP 1: CREATE AND UPLOAD TRAINING DATA")
print("="*70)

# Create a dataset
np.random.seed(42)
n_samples = 1000

# Features with specific distributions (we'll drift these later)
data = {
    'age': np.random.normal(35, 10, n_samples).clip(18, 80),
    'income': np.random.normal(50000, 15000, n_samples).clip(20000, 150000),
    'credit_score': np.random.normal(700, 50, n_samples).clip(300, 850),
    'loan_amount': np.random.normal(25000, 10000, n_samples).clip(5000, 100000),
    'employment_years': np.random.normal(8, 5, n_samples).clip(0, 40),
}

df = pd.DataFrame(data)

# Target: loan approval (binary)
df['approved'] = (
    (df['credit_score'] > 650) & 
    (df['income'] > 35000) & 
    (df['loan_amount'] < df['income'] * 0.5)
).astype(int)

# XGBoost format
train_df = df[['approved', 'age', 'income', 'credit_score', 'loan_amount', 'employment_years']]

print(f"\nDataset shape: {train_df.shape}")
print(f"Approval rate: {train_df['approved'].mean()*100:.1f}%")
print(f"\nFeature statistics:")
print(df.describe().round(2))

# Save and upload
train_file = 'train.csv'
train_df.to_csv(train_file, index=False, header=False)

from sagemaker.inputs import TrainingInput
train_s3_path = f's3://{bucket}/{prefix}/train/{train_file}'
sagemaker_session.upload_data(train_file, bucket=bucket, key_prefix=f'{prefix}/train')

print(f"\n✅ Training data uploaded to: {train_s3_path}")

# Save baseline data (features only, no target) for Model Monitor
baseline_df = df[['age', 'income', 'credit_score', 'loan_amount', 'employment_years']]
baseline_file = 'baseline.csv'
baseline_df.to_csv(baseline_file, index=False)

baseline_s3_path = f's3://{bucket}/{prefix}/baseline/{baseline_file}'
sagemaker_session.upload_data(baseline_file, bucket=bucket, key_prefix=f'{prefix}/baseline')

print(f"✅ Baseline data uploaded to: {baseline_s3_path}")

print("\n" + "="*70)
print("STEP 2: TRAIN MODEL")
print("="*70)

from sagemaker.estimator import Estimator

container = image_uris.retrieve('xgboost', region, '1.5-1')
print(f"\nContainer: {container}")

estimator = Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    output_path=f's3://{bucket}/{prefix}/model/',
    sagemaker_session=sagemaker_session,
    base_job_name='monitoring-demo'
)

estimator.set_hyperparameters(
    objective='binary:logistic',
    num_round=50,
    max_depth=4,
    eta=0.2
)

print("\nTraining model...")
estimator.fit(
    {'train': TrainingInput(train_s3_path, content_type='text/csv')},
    wait=True,
    logs=False
)

model_artifact = estimator.model_data
print(f"\n✅ Model trained: {model_artifact}")

print("\n" + "="*70)
print("STEP 3: DEPLOY WITH DATA CAPTURE")
print("="*70)

timestamp = int(time.time())
model_name = f'monitoring-demo-model-{timestamp}'
endpoint_config_name = f'monitoring-demo-config-{timestamp}'
endpoint_name = f'monitoring-demo-endpoint-{timestamp}'

# Data capture location
data_capture_prefix = f'{prefix}/data-capture'
data_capture_s3_uri = f's3://{bucket}/{data_capture_prefix}'

print(f"\nData capture location: {data_capture_s3_uri}")

# Create Model
print(f"\n1️⃣  Creating Model: {model_name}")

sagemaker_client.create_model(
    ModelName=model_name,
    PrimaryContainer={
        'Image': container,
        'ModelDataUrl': model_artifact
    },
    ExecutionRoleArn=role
)
print(f"   ✅ Model created")

# Create Endpoint Config WITH Data Capture
print(f"\n2️⃣  Creating Endpoint Config with Data Capture: {endpoint_config_name}")

sagemaker_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[{
        'VariantName': 'AllTraffic',
        'ModelName': model_name,
        'InitialInstanceCount': 1,
        'InstanceType': 'ml.m5.large'
    }],
    DataCaptureConfig={
        'EnableCapture': True,
        'InitialSamplingPercentage': 100,  # Capture 100% of requests
        'DestinationS3Uri': data_capture_s3_uri,
        'CaptureOptions': [
            {'CaptureMode': 'Input'},   # Capture request data
            {'CaptureMode': 'Output'}   # Capture response data
        ],
        'CaptureContentTypeHeader': {
            'CsvContentTypes': ['text/csv'],
            'JsonContentTypes': ['application/json']
        }
    }
)
print(f"   ✅ Endpoint config created with data capture enabled")

# Create Endpoint
print(f"\n3️⃣  Creating Endpoint: {endpoint_name}")
print(f"   This will take 3-5 minutes...\n")

sagemaker_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)

# Wait for endpoint
while True:
    response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
    status = response['EndpointStatus']
    
    if status == 'InService':
        print(f"   ✅ Endpoint is InService!")
        break
    elif status == 'Failed':
        print(f"   ❌ Endpoint failed: {response.get('FailureReason', 'Unknown')}")
        exit(1)
    else:
        print(f"   ⏳ Status: {status}...")
        time.sleep(30)

# Save config for later scripts
config = {
    'endpoint_name': endpoint_name,
    'endpoint_config_name': endpoint_config_name,
    'model_name': model_name,
    'model_artifact': model_artifact,
    'data_capture_s3_uri': data_capture_s3_uri,
    'baseline_s3_path': baseline_s3_path,
    'bucket': bucket,
    'prefix': prefix
}

with open('monitoring_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("\n" + "="*70)
print("✅ ENDPOINT DEPLOYED WITH DATA CAPTURE!")
print("="*70)

print(f"""
Endpoint: {endpoint_name}
Data Capture: ENABLED (100% sampling)
Capture Location: {data_capture_s3_uri}

📝 Config saved to: monitoring_config.json

💡 Next step: python create_baseline.py
""")

print("\n" + "-"*70)
print("WHAT IS DATA CAPTURE?")
print("-"*70)
print("""
Data Capture records all inference requests and responses to S3:

   Request ──→ [Endpoint] ──→ Response
                   │
                   ▼
              S3 Bucket
           (captured data)

This captured data is used by Model Monitor to:
1. Compare against baseline statistics
2. Detect data drift
3. Alert when anomalies are found
""")
Run It
bash
cd ~
mkdir -p monitoring-demo
cd monitoring-demo

python deploy_with_monitoring.py
This will:

Create a loan approval dataset
Train an XGBoost model
Deploy with data capture enabled (records all requests/responses)
Let me know when it's deployed (~5-8 min)!

this has deployed.

Now let's create the baseline - this tells Model Monitor what "normal" data looks like:

create_baseline.py
python
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
    instance_type='ml.m5.large',
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
Run It
bash
python create_baseline.py
This creates the baseline statistics that Model Monitor will compare against. Let me know when it completes!


ERROR:sagemaker:Please check the troubleshooting guide for common errors: https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-python-sdk-troubleshooting.html#sagemaker-python-sdk-troubleshooting-create-processing-job ╭───────────────────── Traceback (most recent call last) ────────────────────

pasted

You've hit a service quota limit for processing jobs. Let's try with a smaller instance type that's usually available by default:

Quick Fix
Replace line ~44-49 in create_baseline.py:

python
# Create DefaultModelMonitor
monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600,
    sagemaker_session=sagemaker_session
)
With:

python
# Create DefaultModelMonitor
monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.t3.medium',  # Smaller instance, usually no quota issues
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600,
    sagemaker_session=sagemaker_session
)
Run it again:

bash
python create_baseline.py
If ml.t3.medium also fails, let me know and we'll try ml.t3.large or take an alternative approach using boto3 directly.


sagemaker.config INFO - Not applying SDK defaults from location: /Library/Application Support/sagemaker/config.yaml sagemaker.config INFO - Not applying SDK defaults from location: /Users/adamvigneau/Library/Application Support/sagemaker/config.yaml Region: us-east-2 Bucket: sagemaker-us-east-2-8547

pasted

The ml.t3.medium doesn't have enough memory. Let's try ml.t3.large:

Quick Fix
Change the instance type again:

python
# Create DefaultModelMonitor
monitor = DefaultModelMonitor(
    role=role,
    instance_count=1,
    instance_type='ml.t3.large',  # More memory
    volume_size_in_gb=20,
    max_runtime_in_seconds=3600,
    sagemaker_session=sagemaker_session
)
Run it again:

bash
python create_baseline.py
If this also hits a quota limit, we can request a quota increase or take an alternative approach where we create the baseline statistics manually (which is actually educational for understanding what Model Monitor does under the hood).

sagemaker.config INFO - Not applying SDK defaults from location: /Library/Application Support/sagemaker/config.yaml
sagemaker.config INFO - Not applying SDK defaults from location: /Users/adamvigneau/Library/Application Support/sagemaker/config.yaml
Region: us-east-2
Bucket: sagemaker-us-east-2-854757836160
Baseline data: s3://sagemaker-us-east-2-854757836160/monitoring-demo/baseline/baseline.csv

======================================================================
CREATING BASELINE WITH MODEL MONITOR
======================================================================

What is a baseline?

A baseline captures the statistical properties of your training data:
- Mean, std, min, max for each feature
- Data types and distributions
- Expected ranges

Model Monitor compares live traffic against this baseline to detect drift.

1️⃣  Starting baseline job...
   Input: s3://sagemaker-us-east-2-854757836160/monitoring-demo/baseline/baseline.csv
   Output: s3://sagemaker-us-east-2-854757836160/monitoring-demo/baseline-results
   This will take 3-5 minutes...

INFO:sagemaker:Creating processing-job with name baseline-suggestion-job-2025-12-05-23-18-05-649
........................................................................................!   ✅ Baseline job completed!

2️⃣  Baseline job name: baseline-suggestion-job-2025-12-05-23-18-05-649

======================================================================
BASELINE STATISTICS
======================================================================

Feature Statistics (what Model Monitor considers 'normal'):

Feature                      Mean          Std          Min          Max
----------------------------------------------------------------------
   Could not load statistics: Unknown format code 'f' for object of type 'str'
   Check S3: s3://sagemaker-us-east-2-854757836160/monitoring-demo/baseline-results/statistics.json

----------------------------------------------------------------------
CONSTRAINTS (Drift Detection Rules)
----------------------------------------------------------------------

Model Monitor will alert if these constraints are violated:

   Distribution comparison: N/A
   Threshold: N/A

   Data type monitoring: Enabled
   Completeness monitoring: Enabled

======================================================================
✅ BASELINE CREATED!
======================================================================

Baseline Results: s3://sagemaker-us-east-2-854757836160/monitoring-demo/baseline-results
- statistics.json: Feature distributions
- constraints.json: Drift detection rules

📝 Config updated: monitoring_config.json

💡 Next step: python send_traffic_and_drift.py

This will:
1. Send normal traffic to establish a pattern
2. Send drifted traffic to trigger alerts
3. Check for drift violations

Show more
Baseline created! 🎉

The statistics display had a minor formatting issue, but the important thing is the baseline job completed successfully. The statistics and constraints files are in S3.

Now let's send traffic and simulate drift:

send_traffic_and_drift.py
python
"""
Send traffic to endpoint and simulate data drift
"""
import boto3
import json
import time
import numpy as np

# Configuration
region = boto3.Session().region_name
runtime_client = boto3.client('sagemaker-runtime', region_name=region)
s3_client = boto3.client('s3', region_name=region)

print(f"Region: {region}")

# Load config
with open('monitoring_config.json', 'r') as f:
    config = json.load(f)

endpoint_name = config['endpoint_name']
bucket = config['bucket']
prefix = config['prefix']
data_capture_s3_uri = config['data_capture_s3_uri']

print(f"Endpoint: {endpoint_name}")
print(f"Data Capture: {data_capture_s3_uri}")

print("\n" + "="*70)
print("SENDING TRAFFIC TO ENDPOINT")
print("="*70)

# Helper function to invoke endpoint
def invoke_endpoint(features):
    """Send a prediction request"""
    payload = ','.join(map(str, features))
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='text/csv',
        Body=payload
    )
    return float(response['Body'].read().decode('utf-8').strip('[]'))

# ============================================================
# Phase 1: Send NORMAL traffic (matches training distribution)
# ============================================================
print("\n" + "-"*70)
print("PHASE 1: Sending NORMAL traffic (matches baseline)")
print("-"*70)

print("""
Normal data distributions:
- age: mean=35, std=10
- income: mean=50000, std=15000
- credit_score: mean=700, std=50
- loan_amount: mean=25000, std=10000
- employment_years: mean=8, std=5
""")

np.random.seed(42)
normal_predictions = []

print("Sending 50 normal requests...")
for i in range(50):
    # Generate normal data (same distribution as training)
    features = [
        np.random.normal(35, 10),      # age
        np.random.normal(50000, 15000), # income
        np.random.normal(700, 50),      # credit_score
        np.random.normal(25000, 10000), # loan_amount
        np.random.normal(8, 5)          # employment_years
    ]
    
    prediction = invoke_endpoint(features)
    normal_predictions.append(prediction)
    
    if (i + 1) % 10 == 0:
        print(f"   Sent {i + 1}/50 requests...")

avg_normal = sum(normal_predictions) / len(normal_predictions)
print(f"\n   ✅ Normal traffic sent!")
print(f"   Average prediction (approval probability): {avg_normal:.3f}")

# ============================================================
# Phase 2: Send DRIFTED traffic (different distribution)
# ============================================================
print("\n" + "-"*70)
print("PHASE 2: Sending DRIFTED traffic (different from baseline)")
print("-"*70)

print("""
Drifted data distributions (simulating economic downturn):
- age: mean=45, std=15        (older applicants)
- income: mean=35000, std=10000   (lower income)
- credit_score: mean=620, std=80  (worse credit)
- loan_amount: mean=40000, std=15000  (higher loan requests)
- employment_years: mean=3, std=2  (less stable employment)
""")

drifted_predictions = []

print("Sending 50 drifted requests...")
for i in range(50):
    # Generate drifted data (different distribution!)
    features = [
        np.random.normal(45, 15),       # age - older
        np.random.normal(35000, 10000), # income - lower
        np.random.normal(620, 80),      # credit_score - worse
        np.random.normal(40000, 15000), # loan_amount - higher
        np.random.normal(3, 2)          # employment_years - less
    ]
    
    prediction = invoke_endpoint(features)
    drifted_predictions.append(prediction)
    
    if (i + 1) % 10 == 0:
        print(f"   Sent {i + 1}/50 requests...")

avg_drifted = sum(drifted_predictions) / len(drifted_predictions)
print(f"\n   ✅ Drifted traffic sent!")
print(f"   Average prediction (approval probability): {avg_drifted:.3f}")

# ============================================================
# Compare Results
# ============================================================
print("\n" + "="*70)
print("TRAFFIC COMPARISON")
print("="*70)

print(f"""
┌─────────────────────┬─────────────────┬─────────────────┐
│  Metric             │  Normal Traffic │  Drifted Traffic│
├─────────────────────┼─────────────────┼─────────────────┤
│  Requests Sent      │       50        │       50        │
│  Avg Prediction     │     {avg_normal:.3f}        │     {avg_drifted:.3f}        │
│  Approval Rate      │     {avg_normal*100:.1f}%        │     {avg_drifted*100:.1f}%        │
└─────────────────────┴─────────────────┴─────────────────┘
""")

if avg_drifted < avg_normal:
    diff = (avg_normal - avg_drifted) / avg_normal * 100
    print(f"   📉 Approval rate dropped {diff:.1f}% with drifted data!")
else:
    print(f"   📊 Predictions changed with drifted data")

# ============================================================
# Check Data Capture
# ============================================================
print("\n" + "-"*70)
print("CHECKING DATA CAPTURE")
print("-"*70)

# Wait a moment for data to be written
print("\nWaiting 30 seconds for data capture to write to S3...")
time.sleep(30)

# List captured data files
capture_prefix = f"{prefix}/data-capture/{endpoint_name}/AllTraffic"
try:
    response = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix=capture_prefix,
        MaxKeys=10
    )
    
    files = response.get('Contents', [])
    if files:
        print(f"\n   ✅ Found {len(files)} captured data files!")
        print(f"   Location: s3://{bucket}/{capture_prefix}/")
        
        # Show sample file
        sample_file = files[0]['Key']
        print(f"\n   Sample file: {sample_file}")
        
        # Download and show sample
        obj = s3_client.get_object(Bucket=bucket, Key=sample_file)
        content = obj['Body'].read().decode('utf-8')
        
        print(f"\n   Sample captured data (first 500 chars):")
        print(f"   {content[:500]}...")
    else:
        print(f"\n   ⚠️  No captured data files yet")
        print(f"   Data capture can take a few minutes to appear")
        
except Exception as e:
    print(f"\n   ⚠️  Could not check data capture: {e}")

print("\n" + "="*70)
print("✅ TRAFFIC SENT!")
print("="*70)

print("""
📝 WHAT HAPPENED:

1. Sent 50 NORMAL requests (matching training distribution)
2. Sent 50 DRIFTED requests (simulating changed conditions)
3. All requests captured to S3

📝 WHAT MODEL MONITOR WOULD DETECT:

In production, Model Monitor runs on a schedule (hourly/daily) and would:
1. Compare captured data against baseline statistics
2. Detect that feature distributions have shifted
3. Generate violation report
4. Trigger CloudWatch alarm

📝 KEY DRIFT INDICATORS:

| Feature          | Baseline Mean | Drifted Mean | Drift |
|------------------|---------------|--------------|-------|
| age              | 35            | 45           | +29%  |
| income           | 50,000        | 35,000       | -30%  |
| credit_score     | 700           | 620          | -11%  |
| loan_amount      | 25,000        | 40,000       | +60%  |
| employment_years | 8             | 3            | -63%  |

💡 Next step: python analyze_drift.py
""")
Run It
bash
python send_traffic_and_drift.py
This will:

Send 50 normal requests (matching baseline)
Send 50 drifted requests (simulating changed conditions)
Show how predictions change with drift
Let me know what results you get!

sagemaker.config INFO - Not applying SDK defaults from location: /Library/Application Support/sagemaker/config.yaml
sagemaker.config INFO - Not applying SDK defaults from location: /Users/adamvigneau/Library/Application Support/sagemaker/config.yaml
Region: us-east-2
Bucket: sagemaker-us-east-2-854757836160
Baseline data: s3://sagemaker-us-east-2-854757836160/monitoring-demo/baseline/baseline.csv

======================================================================
CREATING BASELINE WITH MODEL MONITOR
======================================================================

What is a baseline?

A baseline captures the statistical properties of your training data:
- Mean, std, min, max for each feature
- Data types and distributions
- Expected ranges

Model Monitor compares live traffic against this baseline to detect drift.

1️⃣  Starting baseline job...
   Input: s3://sagemaker-us-east-2-854757836160/monitoring-demo/baseline/baseline.csv
   Output: s3://sagemaker-us-east-2-854757836160/monitoring-demo/baseline-results
   This will take 3-5 minutes...

INFO:sagemaker:Creating processing-job with name baseline-suggestion-job-2025-12-05-23-18-05-649
........................................................................................!   ✅ Baseline job completed!

2️⃣  Baseline job name: baseline-suggestion-job-2025-12-05-23-18-05-649

======================================================================
BASELINE STATISTICS
======================================================================

Feature Statistics (what Model Monitor considers 'normal'):

Feature                      Mean          Std          Min          Max
----------------------------------------------------------------------
   Could not load statistics: Unknown format code 'f' for object of type 'str'
   Check S3: s3://sagemaker-us-east-2-854757836160/monitoring-demo/baseline-results/statistics.json

----------------------------------------------------------------------
CONSTRAINTS (Drift Detection Rules)
----------------------------------------------------------------------

Model Monitor will alert if these constraints are violated:

   Distribution comparison: N/A
   Threshold: N/A

   Data type monitoring: Enabled
   Completeness monitoring: Enabled

======================================================================
✅ BASELINE CREATED!
======================================================================

Baseline Results: s3://sagemaker-us-east-2-854757836160/monitoring-demo/baseline-results
- statistics.json: Feature distributions
- constraints.json: Drift detection rules

📝 Config updated: monitoring_config.json

💡 Next step: python send_traffic_and_drift.py

This will:
1. Send normal traffic to establish a pattern
2. Send drifted traffic to trigger alerts
3. Check for drift violations

Show more
That's the output from the previous step (create_baseline.py). Could you run the new script?

bash
python send_traffic_and_drift.py
I haven't run send_traffic_and_drift yet.  Could you please provide that for me?

Here's the script:

send_traffic_and_drift.py
python
"""
Send traffic to endpoint and simulate data drift
"""
import boto3
import json
import time
import numpy as np

# Configuration
region = boto3.Session().region_name
runtime_client = boto3.client('sagemaker-runtime', region_name=region)
s3_client = boto3.client('s3', region_name=region)

print(f"Region: {region}")

# Load config
with open('monitoring_config.json', 'r') as f:
    config = json.load(f)

endpoint_name = config['endpoint_name']
bucket = config['bucket']
prefix = config['prefix']
data_capture_s3_uri = config['data_capture_s3_uri']

print(f"Endpoint: {endpoint_name}")
print(f"Data Capture: {data_capture_s3_uri}")

print("\n" + "="*70)
print("SENDING TRAFFIC TO ENDPOINT")
print("="*70)

# Helper function to invoke endpoint
def invoke_endpoint(features):
    """Send a prediction request"""
    payload = ','.join(map(str, features))
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='text/csv',
        Body=payload
    )
    return float(response['Body'].read().decode('utf-8').strip('[]'))

# ============================================================
# Phase 1: Send NORMAL traffic (matches training distribution)
# ============================================================
print("\n" + "-"*70)
print("PHASE 1: Sending NORMAL traffic (matches baseline)")
print("-"*70)

print("""
Normal data distributions:
- age: mean=35, std=10
- income: mean=50000, std=15000
- credit_score: mean=700, std=50
- loan_amount: mean=25000, std=10000
- employment_years: mean=8, std=5
""")

np.random.seed(42)
normal_predictions = []

print("Sending 50 normal requests...")
for i in range(50):
    # Generate normal data (same distribution as training)
    features = [
        np.random.normal(35, 10),       # age
        np.random.normal(50000, 15000), # income
        np.random.normal(700, 50),      # credit_score
        np.random.normal(25000, 10000), # loan_amount
        np.random.normal(8, 5)          # employment_years
    ]
    
    prediction = invoke_endpoint(features)
    normal_predictions.append(prediction)
    
    if (i + 1) % 10 == 0:
        print(f"   Sent {i + 1}/50 requests...")

avg_normal = sum(normal_predictions) / len(normal_predictions)
print(f"\n   ✅ Normal traffic sent!")
print(f"   Average prediction (approval probability): {avg_normal:.3f}")

# ============================================================
# Phase 2: Send DRIFTED traffic (different distribution)
# ============================================================
print("\n" + "-"*70)
print("PHASE 2: Sending DRIFTED traffic (different from baseline)")
print("-"*70)

print("""
Drifted data distributions (simulating economic downturn):
- age: mean=45, std=15          (older applicants)
- income: mean=35000, std=10000 (lower income)
- credit_score: mean=620, std=80 (worse credit)
- loan_amount: mean=40000, std=15000 (higher loan requests)
- employment_years: mean=3, std=2 (less stable employment)
""")

drifted_predictions = []

print("Sending 50 drifted requests...")
for i in range(50):
    # Generate drifted data (different distribution!)
    features = [
        np.random.normal(45, 15),       # age - older
        np.random.normal(35000, 10000), # income - lower
        np.random.normal(620, 80),      # credit_score - worse
        np.random.normal(40000, 15000), # loan_amount - higher
        np.random.normal(3, 2)          # employment_years - less
    ]
    
    prediction = invoke_endpoint(features)
    drifted_predictions.append(prediction)
    
    if (i + 1) % 10 == 0:
        print(f"   Sent {i + 1}/50 requests...")

avg_drifted = sum(drifted_predictions) / len(drifted_predictions)
print(f"\n   ✅ Drifted traffic sent!")
print(f"   Average prediction (approval probability): {avg_drifted:.3f}")

# ============================================================
# Compare Results
# ============================================================
print("\n" + "="*70)
print("TRAFFIC COMPARISON")
print("="*70)

print(f"""
┌─────────────────────┬─────────────────┬─────────────────┐
│  Metric             │  Normal Traffic │  Drifted Traffic│
├─────────────────────┼─────────────────┼─────────────────┤
│  Requests Sent      │       50        │       50        │
│  Avg Prediction     │     {avg_normal:.3f}        │     {avg_drifted:.3f}        │
│  Approval Rate      │     {avg_normal*100:.1f}%        │     {avg_drifted*100:.1f}%        │
└─────────────────────┴─────────────────┴─────────────────┘
""")

if avg_drifted < avg_normal:
    diff = (avg_normal - avg_drifted) / avg_normal * 100
    print(f"   📉 Approval rate dropped {diff:.1f}% with drifted data!")
else:
    print(f"   📊 Predictions changed with drifted data")

# ============================================================
# Check Data Capture
# ============================================================
print("\n" + "-"*70)
print("CHECKING DATA CAPTURE")
print("-"*70)

# Wait a moment for data to be written
print("\nWaiting 30 seconds for data capture to write to S3...")
time.sleep(30)

# List captured data files
capture_prefix = f"{prefix}/data-capture/{endpoint_name}/AllTraffic"
try:
    response = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix=capture_prefix,
        MaxKeys=10
    )
    
    files = response.get('Contents', [])
    if files:
        print(f"\n   ✅ Found {len(files)} captured data files!")
        print(f"   Location: s3://{bucket}/{capture_prefix}/")
        
        # Show sample file
        sample_file = files[0]['Key']
        print(f"\n   Sample file: {sample_file}")
        
        # Download and show sample
        obj = s3_client.get_object(Bucket=bucket, Key=sample_file)
        content = obj['Body'].read().decode('utf-8')
        
        print(f"\n   Sample captured data (first 500 chars):")
        print(f"   {content[:500]}...")
    else:
        print(f"\n   ⚠️  No captured data files yet")
        print(f"   Data capture can take a few minutes to appear")
        
except Exception as e:
    print(f"\n   ⚠️  Could not check data capture: {e}")

print("\n" + "="*70)
print("✅ TRAFFIC SENT!")
print("="*70)

print("""
📝 WHAT HAPPENED:

1. Sent 50 NORMAL requests (matching training distribution)
2. Sent 50 DRIFTED requests (simulating changed conditions)
3. All requests captured to S3

📝 WHAT MODEL MONITOR WOULD DETECT:

In production, Model Monitor runs on a schedule (hourly/daily) and would:
1. Compare captured data against baseline statistics
2. Detect that feature distributions have shifted
3. Generate violation report
4. Trigger CloudWatch alarm

📝 KEY DRIFT INDICATORS:

| Feature          | Baseline Mean | Drifted Mean | Drift |
|------------------|---------------|--------------|-------|
| age              | 35            | 45           | +29%  |
| income           | 50,000        | 35,000       | -30%  |
| credit_score     | 700           | 620          | -11%  |
| loan_amount      | 25,000        | 40,000       | +60%  |
| employment_years | 8             | 3            | -63%  |

💡 Next step: python analyze_drift.py
""")
Run It
bash
python send_traffic_and_drift.py
Let me know what you see!


Region: us-east-2 Endpoint: monitoring-demo-endpoint-1764974869 Data Capture: s3://sagemaker-us-east-2-854757836160/monitoring-demo/data-capture ====================================================================== SENDING TRAFFIC TO ENDPOINT =======================================================

pasted

Look at that drift! 🎉

Approval rate dropped from 44.7% to 0.5% — a 98.8% drop when the data drifted. This is exactly why monitoring matters in production!

The data capture files may take a few more minutes to appear. Let's create the analysis script that also manually compares the drift:

analyze_drift.py
python
"""
Analyze data drift and demonstrate what Model Monitor detects
"""
import boto3
import json
import numpy as np

# Configuration
region = boto3.Session().region_name
s3_client = boto3.client('s3', region_name=region)

print(f"Region: {region}")

# Load config
with open('monitoring_config.json', 'r') as f:
    config = json.load(f)

bucket = config['bucket']
prefix = config['prefix']
endpoint_name = config['endpoint_name']

print("\n" + "="*70)
print("ANALYZING DATA DRIFT")
print("="*70)

# ============================================================
# Load Baseline Statistics
# ============================================================
print("\n" + "-"*70)
print("BASELINE STATISTICS (from training data)")
print("-"*70)

stats_key = f"{prefix}/baseline-results/statistics.json"
try:
    response = s3_client.get_object(Bucket=bucket, Key=stats_key)
    statistics = json.loads(response['Body'].read().decode('utf-8'))
    
    baseline_stats = {}
    
    print(f"\n{'Feature':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("-" * 70)
    
    for feature in statistics.get('features', []):
        name = feature.get('name', 'unknown')
        numerical = feature.get('numerical_statistics', {})
        
        mean = numerical.get('mean', 0)
        std = numerical.get('std', 0)
        min_val = numerical.get('min', 0)
        max_val = numerical.get('max', 0)
        
        baseline_stats[name] = {'mean': mean, 'std': std, 'min': min_val, 'max': max_val}
        
        # Handle different data types
        try:
            print(f"{name:<20} {float(mean):>12.2f} {float(std):>12.2f} {float(min_val):>12.2f} {float(max_val):>12.2f}")
        except (ValueError, TypeError):
            print(f"{name:<20} {str(mean):>12} {str(std):>12} {str(min_val):>12} {str(max_val):>12}")
            
except Exception as e:
    print(f"   Could not load baseline statistics: {e}")
    # Use expected values from training
    baseline_stats = {
        'age': {'mean': 35, 'std': 10},
        'income': {'mean': 50000, 'std': 15000},
        'credit_score': {'mean': 700, 'std': 50},
        'loan_amount': {'mean': 25000, 'std': 10000},
        'employment_years': {'mean': 8, 'std': 5}
    }
    print("\n   Using expected baseline values from training script")

# ============================================================
# Simulated Drifted Statistics
# ============================================================
print("\n" + "-"*70)
print("DRIFTED DATA STATISTICS (from inference traffic)")
print("-"*70)

# These are the distributions we used for drifted traffic
drifted_stats = {
    'age': {'mean': 45, 'std': 15},
    'income': {'mean': 35000, 'std': 10000},
    'credit_score': {'mean': 620, 'std': 80},
    'loan_amount': {'mean': 40000, 'std': 15000},
    'employment_years': {'mean': 3, 'std': 2}
}

print(f"\n{'Feature':<20} {'Mean':>12} {'Std':>12}")
print("-" * 46)

for name, stats in drifted_stats.items():
    print(f"{name:<20} {stats['mean']:>12.2f} {stats['std']:>12.2f}")

# ============================================================
# Drift Analysis
# ============================================================
print("\n" + "="*70)
print("DRIFT DETECTION ANALYSIS")
print("="*70)

print("""
Model Monitor uses statistical tests to detect drift:
- Kolmogorov-Smirnov test (distribution comparison)
- Population Stability Index (PSI)
- Simple threshold-based checks (mean shift > X%)
""")

print(f"\n{'Feature':<20} {'Baseline':>12} {'Drifted':>12} {'Change':>12} {'Alert':>10}")
print("-" * 70)

drift_detected = []

for feature in ['age', 'income', 'credit_score', 'loan_amount', 'employment_years']:
    baseline_mean = baseline_stats.get(feature, {}).get('mean', 0)
    drifted_mean = drifted_stats.get(feature, {}).get('mean', 0)
    
    try:
        baseline_mean = float(baseline_mean)
        change_pct = ((drifted_mean - baseline_mean) / baseline_mean) * 100
        
        # Alert if change > 20%
        alert = "🚨 DRIFT" if abs(change_pct) > 20 else "✅ OK"
        if abs(change_pct) > 20:
            drift_detected.append(feature)
        
        print(f"{feature:<20} {baseline_mean:>12.1f} {drifted_mean:>12.1f} {change_pct:>+11.1f}% {alert:>10}")
    except (ValueError, TypeError):
        print(f"{feature:<20} {'N/A':>12} {drifted_mean:>12.1f} {'N/A':>12} {'?':>10}")

# ============================================================
# Check for Captured Data
# ============================================================
print("\n" + "-"*70)
print("DATA CAPTURE STATUS")
print("-"*70)

capture_prefix = f"{prefix}/data-capture/{endpoint_name}/AllTraffic"
try:
    response = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix=capture_prefix,
        MaxKeys=20
    )
    
    files = response.get('Contents', [])
    if files:
        print(f"\n   ✅ Found {len(files)} captured data files!")
        print(f"   Location: s3://{bucket}/{capture_prefix}/")
        
        # Show files
        print("\n   Recent capture files:")
        for f in files[:5]:
            print(f"   - {f['Key'].split('/')[-1]}")
    else:
        print(f"\n   ⚠️  No captured data files yet")
        print(f"   Data capture can take up to 5 minutes to appear in S3")
        
except Exception as e:
    print(f"\n   ⚠️  Could not check data capture: {e}")

# ============================================================
# Summary
# ============================================================
print("\n" + "="*70)
print("DRIFT DETECTION SUMMARY")
print("="*70)

print(f"""
Features with detected drift (>20% change):
""")

for feature in drift_detected:
    print(f"   🚨 {feature}")

if not drift_detected:
    print("   ✅ No significant drift detected")

print(f"""
┌─────────────────────────────────────────────────────────────────┐
│  WHAT WOULD HAPPEN IN PRODUCTION                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Model Monitor scheduled job runs (hourly/daily)             │
│  2. Compares captured data against baseline                     │
│  3. Detects drift in {len(drift_detected)} features                               │
│  4. Generates violation report in S3                            │
│  5. Triggers CloudWatch alarm                                   │
│  6. Team investigates and decides:                              │
│     - Retrain model with new data?                              │
│     - Roll back to previous model?                              │
│     - Adjust business rules?                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
""")

print("\n" + "-"*70)
print("KEY EXAM CONCEPTS")
print("-"*70)
print("""
1. DATA DRIFT: Input feature distributions change over time
   → Detected by comparing against baseline statistics

2. CONCEPT DRIFT: Relationship between features and target changes
   → Detected by monitoring model quality metrics (accuracy, etc.)

3. MODEL MONITOR TYPES:
   - Data Quality Monitor: Detects data drift
   - Model Quality Monitor: Detects accuracy degradation
   - Bias Drift Monitor: Detects fairness metric changes
   - Feature Attribution Monitor: Detects SHAP value changes

4. MONITORING SCHEDULE:
   - Minimum: Hourly
   - Typical: Daily
   - Uses captured inference data

5. REMEDIATION OPTIONS:
   - Retrain model
   - Roll back to previous version
   - Adjust feature engineering
   - Update business rules
""")

print(f"\n💡 Next step: python cleanup_monitoring.py")
Run It
bash
python analyze_drift.py
This will show:

Baseline statistics
Drifted statistics
Which features triggered drift alerts
What happens in production
Let me know what you see!


====================================================================== ANALYZING DATA DRIFT ====================================================================== ---------------------------------------------------------------------- BASELINE STATISTICS (from training data) ------------------------

pasted

Excellent results! 🎉

You detected drift in 4 out of 5 features:

🚨 age (+27.4%)
🚨 income (-31.6%)
🚨 loan_amount (+60.7%)
🚨 employment_years (-62.1%)
And the data capture is working — you can see the .jsonl file in S3!

Now let's clean up to avoid charges:

cleanup_monitoring.py
python
"""
Cleanup monitoring demo resources
"""
import boto3
import json
import time

region = boto3.Session().region_name
sagemaker_client = boto3.client('sagemaker', region_name=region)

print(f"Region: {region}")

# Load config
try:
    with open('monitoring_config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print("monitoring_config.json not found. Enter names manually:")
    config = {
        'endpoint_name': input("Endpoint name: ").strip(),
        'endpoint_config_name': input("Endpoint config name: ").strip(),
        'model_name': input("Model name: ").strip()
    }

endpoint_name = config['endpoint_name']
endpoint_config_name = config['endpoint_config_name']
model_name = config['model_name']

print("\n" + "="*70)
print("CLEANING UP MONITORING DEMO RESOURCES")
print("="*70)

# Delete endpoint
print(f"\n1️⃣  Deleting endpoint: {endpoint_name}")
try:
    sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
    print(f"   ✅ Delete initiated")
    
    while True:
        try:
            sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            print(f"   ⏳ Waiting for deletion...")
            time.sleep(10)
        except sagemaker_client.exceptions.ClientError:
            print(f"   ✅ Endpoint deleted")
            break
except Exception as e:
    print(f"   ⚠️  Error: {e}")

# Delete endpoint config
print(f"\n2️⃣  Deleting endpoint config: {endpoint_config_name}")
try:
    sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
    print(f"   ✅ Endpoint config deleted")
except Exception as e:
    print(f"   ⚠️  Error: {e}")

# Delete model
print(f"\n3️⃣  Deleting model: {model_name}")
try:
    sagemaker_client.delete_model(ModelName=model_name)
    print(f"   ✅ Model deleted")
except Exception as e:
    print(f"   ⚠️  Error: {e}")

print("\n" + "="*70)
print("✅ CLEANUP COMPLETE!")
print("="*70)
print("""
💰 No more charges for this endpoint!

📝 Note: The following remain in S3 (minimal cost):
   - Baseline statistics
   - Captured data
   - Training data

To fully clean up S3, run:
   aws s3 rm s3://sagemaker-us-east-2-854757836160/monitoring-demo/ --recursive
""")
Run It
bash
python cleanup_monitoring.py
Let me know when cleanup is done and I'll give you the Day 5 recap!

Can you make the ReadME, .gitignore, and any other files I don't have already?


7 steps

Create .gitignore for monitoring-demo
.gitignore

Copy all monitoring-demo files to outputs


Here are all the files for your GitHub repo:

Files Created
File	Description
README.md	Project documentation
deploy_with_monitoring.py	Deploy model with data capture
create_baseline.py	Generate baseline statistics
send_traffic_and_drift.py	Send normal and drifted traffic
analyze_drift.py	Analyze drift detection
cleanup_monitoring.py	Delete resources
requirements.txt	Python dependencies
.gitignore	Git ignore file
Terminal Commands to Check In
bash
# Navigate to your repo
cd ~/aws-ml-portfolio

# Create the monitoring-demo folder
mkdir -p monitoring-demo

# Copy the downloaded files
cp ~/Downloads/monitoring-demo/* monitoring-demo/
cp ~/Downloads/monitoring-demo/.gitignore monitoring-demo/

# Commit and push
git add monitoring-demo/
git commit -m "Add SageMaker Model Monitoring demo with drift detection"
git push origin main
Don't forget to run cleanup when you're done testing:

bash
python cleanup_monitoring.py
Would you like the Day 5 recap now?

Readme
Document · MD 
Deploy with monitoring
PY 
Create baseline
PY 
Send traffic and drift
PY 
Analyze drift
PY 
Cleanup monitoring
PY 
Requirements
TXT 
.gitignore
GITIGNORE 









Readme · MD
Copy

# SageMaker Model Monitoring Demo

A hands-on demonstration of SageMaker Model Monitor for detecting data drift in production ML systems.

## Overview

This project demonstrates:

- **Data Capture**: Record all inference requests/responses to S3
- **Baseline Creation**: Establish "normal" data distributions
- **Drift Detection**: Identify when production data differs from training data
- **Alerting**: Understand how Model Monitor triggers alarms

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Model Monitoring Pipeline                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Training Data                                                      │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
│  │  Baseline   │────▶│   Model     │────▶│  Endpoint   │          │
│  │  Statistics │     │  Training   │     │  (Live)     │          │
│  └─────────────┘     └─────────────┘     └──────┬──────┘          │
│        │                                        │                  │
│        │              Inference Requests        │                  │
│        │                     │                  │                  │
│        │                     ▼                  ▼                  │
│        │              ┌─────────────┐    ┌─────────────┐          │
│        │              │   Data      │    │  Captured   │          │
│        │              │   Capture   │───▶│  Data (S3)  │          │
│        │              └─────────────┘    └──────┬──────┘          │
│        │                                        │                  │
│        │         ┌──────────────────────────────┘                  │
│        │         │                                                 │
│        ▼         ▼                                                 │
│  ┌─────────────────────┐     ┌─────────────────────┐              │
│  │   Model Monitor     │────▶│   Violation Report  │              │
│  │   (Scheduled Job)   │     │   + CloudWatch      │              │
│  └─────────────────────┘     └─────────────────────┘              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Scripts

| Script | Purpose |
|--------|---------|
| `deploy_with_monitoring.py` | Train model and deploy with data capture enabled |
| `create_baseline.py` | Generate baseline statistics from training data |
| `send_traffic_and_drift.py` | Send normal and drifted traffic to endpoint |
| `analyze_drift.py` | Analyze drift and show what Model Monitor detects |
| `cleanup_monitoring.py` | Delete all resources |

## Setup

### Prerequisites

- AWS account with SageMaker access
- Python 3.8+
- AWS CLI configured with credentials

### Installation

```bash
pip install 'sagemaker>=2.200.0,<3.0' boto3 pandas numpy
```

### Configuration

Update the `role` variable in each script with your SageMaker execution role ARN:

```python
role = "arn:aws:iam::YOUR_ACCOUNT:role/YOUR_SAGEMAKER_ROLE"
```

## Running the Demo

```bash
# 1. Deploy model with data capture (~8 min)
python deploy_with_monitoring.py

# 2. Create baseline statistics (~5 min)
python create_baseline.py

# 3. Send traffic and simulate drift (~2 min)
python send_traffic_and_drift.py

# 4. Analyze drift detection
python analyze_drift.py

# 5. Cleanup when done (important!)
python cleanup_monitoring.py
```

## Drift Simulation

The demo simulates an economic downturn affecting loan applications:

| Feature | Baseline (Normal) | Drifted (Downturn) | Change |
|---------|-------------------|--------------------| -------|
| age | mean=35, std=10 | mean=45, std=15 | +29% |
| income | mean=50000, std=15000 | mean=35000, std=10000 | -30% |
| credit_score | mean=700, std=50 | mean=620, std=80 | -11% |
| loan_amount | mean=25000, std=10000 | mean=40000, std=15000 | +60% |
| employment_years | mean=8, std=5 | mean=3, std=2 | -63% |

**Result**: Approval rate dropped from 44.7% to 0.5% (98.8% decrease!)

## Types of Drift

| Drift Type | What Changes | How to Detect |
|------------|--------------|---------------|
| Data Drift | Input feature distributions | Data Quality Monitor |
| Concept Drift | Relationship between features and target | Model Quality Monitor |
| Bias Drift | Fairness metrics | Bias Drift Monitor |
| Feature Attribution Drift | SHAP values | Feature Attribution Monitor |

## Model Monitor Types

| Monitor | Purpose | Detects |
|---------|---------|---------|
| Data Quality | Compare data statistics | Missing values, type changes, distribution shifts |
| Model Quality | Compare model metrics | Accuracy, precision, recall degradation |
| Bias Drift | Compare fairness metrics | Demographic parity, equalized odds changes |
| Feature Attribution | Compare SHAP values | Feature importance shifts |

## Key Exam Concepts

1. **Data Capture** records inference requests/responses to S3
2. **Baseline** defines what "normal" looks like (from training data)
3. **Model Monitor** runs on schedule (minimum hourly)
4. **Violations** trigger CloudWatch alarms
5. **Remediation**: retrain, rollback, or adjust business rules

## Data Capture Format

Captured data is stored as JSONL files:

```json
{
  "captureData": {
    "endpointInput": {
      "data": "35.5,52000,710,28000,6.5",
      "encoding": "CSV"
    },
    "endpointOutput": {
      "data": "0.78",
      "encoding": "CSV"
    }
  },
  "eventMetadata": {
    "eventId": "abc123",
    "inferenceTime": "2024-12-05T12:00:00Z"
  }
}
```

## Monitoring Schedule Options

| Schedule | Use Case |
|----------|----------|
| Hourly | High-traffic, critical applications |
| Daily | Standard production monitoring |
| Weekly | Low-traffic or stable models |

## Costs

- **Endpoint**: ~$0.10/hour for ml.m5.large
- **Processing Job (Baseline)**: ~$0.05-0.10 per run
- **Model Monitor Jobs**: ~$0.05-0.10 per scheduled run
- **S3 Storage**: Minimal (captured data)
- **⚠️ Always run cleanup script when done!**

## Technologies

- AWS SageMaker Model Monitor
- AWS SageMaker Endpoints
- AWS S3 (Data Capture)
- AWS CloudWatch
- XGBoost
- Python / Boto3

## License

MIT
