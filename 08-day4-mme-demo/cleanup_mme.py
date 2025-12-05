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