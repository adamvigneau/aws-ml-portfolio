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