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