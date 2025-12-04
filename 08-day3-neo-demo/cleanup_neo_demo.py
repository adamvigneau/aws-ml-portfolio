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