"""
Deploy Neo-compiled model and benchmark inference speed
"""
import boto3
import sagemaker
from sagemaker import image_uris
from sagemaker.model import Model
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
import time
import numpy as np

# ============================================================
# CONFIGURATION - Update role with your SageMaker execution role
# ============================================================
role = "arn:aws:iam::854757836160:role/service-role/AmazonSageMaker-ExecutionRole-20251019T120276"

# Configuration
region = boto3.Session().region_name
sagemaker_session = sagemaker.Session()
sagemaker_client = boto3.client('sagemaker', region_name=region)
runtime_client = boto3.client('sagemaker-runtime', region_name=region)

bucket = sagemaker_session.default_bucket()

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
xgb_container = image_uris.retrieve('xgboost', region, '1.5-1')
neo_container = image_uris.retrieve('xgboost-neo', region, version='latest')

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

original_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name=original_endpoint_name,
    serializer=CSVSerializer(),
    deserializer=JSONDeserializer(),
    wait=True
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

neo_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name=neo_endpoint_name,
    serializer=CSVSerializer(),
    deserializer=JSONDeserializer(),
    wait=True
)

print(f"   ✅ Neo-compiled model deployed!")

# Save endpoint names for cleanup
with open('endpoint_names.txt', 'w') as f:
    f.write(f"{original_endpoint_name}\n")
    f.write(f"{neo_endpoint_name}\n")

print("\n📝 Endpoint names saved to: endpoint_names.txt")

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
print("\n3️⃣  Warming up endpoints...")
for i in range(5):
    invoke_endpoint(original_endpoint_name, to_csv(test_samples[0]))
    invoke_endpoint(neo_endpoint_name, to_csv(test_samples[0]))
print("   ✅ Warm-up complete")

# ============================================================
# Benchmark Original Model
# ============================================================
print("\n4️⃣  Benchmarking ORIGINAL model (100 inferences)...")

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
print("\n5️⃣  Benchmarking NEO-COMPILED model (100 inferences)...")

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
