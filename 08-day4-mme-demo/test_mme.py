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
    # Handle both "[0.123]" and "0.123" formats
    pred_value = prediction.strip('[]')
    results[model_name] = {
        'prediction': float(pred_value),
        'latency': latency
    }

    

    
    
    print(f"   Prediction: {results[model_name]['prediction']:.4f}")
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
    prediction = float(response['Body'].read().decode('utf-8').strip('[]'))
    
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