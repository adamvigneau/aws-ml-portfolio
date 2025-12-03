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
