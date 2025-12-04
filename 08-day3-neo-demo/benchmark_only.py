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