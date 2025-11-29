import boto3
import json
import zipfile
import os

# Load config
with open('mlops_config.json', 'r') as f:
    config = json.load(f)

lambda_client = boto3.client('lambda')
events = boto3.client('events')

print("🚀 Creating Lambda functions...")

# Lambda 1: Trigger Training
print("\n1. Creating Lambda: TriggerTraining...")

lambda1_code = '''
import json
import boto3
import os
from datetime import datetime

sagemaker = boto3.client('sagemaker')

def lambda_handler(event, context):
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    print(f"New data: s3://{bucket}/{key}")
    
    job_name = f"mlops-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    sagemaker.create_training_job(
        TrainingJobName=job_name,
        RoleArn=os.environ['SAGEMAKER_ROLE'],
        AlgorithmSpecification={
            'TrainingImage': '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.5-1',
            'TrainingInputMode': 'File'
        },
        InputDataConfig=[{
            'ChannelName': 'train',
            'DataSource': {'S3DataSource': {
                'S3DataType': 'S3Prefix',
                'S3Uri': f's3://{bucket}/{key}'
            }},
            'ContentType': 'text/csv'
        }],
        OutputDataConfig={
            'S3OutputPath': f"s3://{os.environ['OUTPUT_BUCKET']}/models/"
        },
        ResourceConfig={
            'InstanceType': 'ml.m5.xlarge',
            'InstanceCount': 1,
            'VolumeSizeInGB': 30
        },
        StoppingCondition={'MaxRuntimeInSeconds': 3600},
        HyperParameters={
            'objective': 'binary:logistic',
            'num_round': '100',
            'max_depth': '5',
            'eta': '0.2'
        }
    )
    
    print(f"Training started: {job_name}")
    return {'statusCode': 200, 'body': job_name}
'''

with open('/tmp/lambda1.py', 'w') as f:
    f.write(lambda1_code)

with zipfile.ZipFile('/tmp/lambda1.zip', 'w') as z:
    z.write('/tmp/lambda1.py', 'lambda_function.py')

with open('/tmp/lambda1.zip', 'rb') as f:
    try:
        lambda1 = lambda_client.create_function(
            FunctionName='MLOps-TriggerTraining',
            Runtime='python3.11',
            Role=config['lambda_role_arn'],
            Handler='lambda_function.lambda_handler',
            Code={'ZipFile': f.read()},
            Timeout=300,
            Environment={'Variables': {
                'SAGEMAKER_ROLE': config['sagemaker_role_arn'],
                'OUTPUT_BUCKET': config['output_bucket']
            }}
        )
        lambda1_arn = lambda1['FunctionArn']
        print(f"✅ Lambda 1 created: {lambda1_arn}")
    except lambda_client.exceptions.ResourceConflictException:
        lambda1_arn = f"arn:aws:lambda:{config['region']}:{config['account_id']}:function:MLOps-TriggerTraining"
        print(f"✅ Lambda 1 already exists: {lambda1_arn}")

# Lambda 2: Register Model
print("\n2. Creating Lambda: RegisterModel...")

lambda2_code = '''
import json
import boto3
import os
from datetime import datetime

sagemaker = boto3.client('sagemaker')
sns = boto3.client('sns')

def lambda_handler(event, context):
    job_name = event['detail']['TrainingJobName']
    status = event['detail']['TrainingJobStatus']
    
    print(f"Training job {job_name}: {status}")
    
    if status != 'Completed':
        return {'statusCode': 400, 'body': 'Job not completed'}
    
    job = sagemaker.describe_training_job(TrainingJobName=job_name)
    model_data = job['ModelArtifacts']['S3ModelArtifacts']
    image = job['AlgorithmSpecification']['TrainingImage']
    
    model_pkg = sagemaker.create_model_package(
        ModelPackageGroupName='mlops-pipeline-models',
        ModelPackageDescription=f'Model from {job_name}',
        InferenceSpecification={
            'Containers': [{'Image': image, 'ModelDataUrl': model_data}],
            'SupportedContentTypes': ['text/csv'],
            'SupportedResponseMIMETypes': ['text/csv']
        },
        ModelApprovalStatus='PendingManualApproval',
        CustomerMetadataProperties={
            'TrainingJobName': job_name,
            'TrainingDate': datetime.now().isoformat()
        }
    )
    
    sns.publish(
        TopicArn=os.environ['SNS_TOPIC'],
        Subject='MLOps: New Model Ready for Approval',
        Message=f"Model: {model_pkg['ModelPackageArn']}\\n\\nJob: {job_name}\\n\\nApprove in SageMaker console"
    )
    
    print(f"Model registered: {model_pkg['ModelPackageArn']}")
    return {'statusCode': 200}
'''

with open('/tmp/lambda2.py', 'w') as f:
    f.write(lambda2_code)

with zipfile.ZipFile('/tmp/lambda2.zip', 'w') as z:
    z.write('/tmp/lambda2.py', 'lambda_function.py')

with open('/tmp/lambda2.zip', 'rb') as f:
    try:
        lambda2 = lambda_client.create_function(
            FunctionName='MLOps-RegisterModel',
            Runtime='python3.11',
            Role=config['lambda_role_arn'],
            Handler='lambda_function.lambda_handler',
            Code={'ZipFile': f.read()},
            Timeout=300,
            Environment={'Variables': {
                'SNS_TOPIC': config['sns_topic_arn']
            }}
        )
        lambda2_arn = lambda2['FunctionArn']
        print(f"✅ Lambda 2 created: {lambda2_arn}")
    except lambda_client.exceptions.ResourceConflictException:
        lambda2_arn = f"arn:aws:lambda:{config['region']}:{config['account_id']}:function:MLOps-RegisterModel"
        print(f"✅ Lambda 2 already exists: {lambda2_arn}")

# Lambda 3: Deploy Model
print("\n3. Creating Lambda: DeployModel...")

lambda3_code = '''
import json
import boto3
import os
from datetime import datetime

sagemaker = boto3.client('sagemaker')
sns = boto3.client('sns')

def lambda_handler(event, context):
    model_pkg_arn = event['detail']['ModelPackageArn']
    status = event['detail']['ModelApprovalStatus']
    
    if status != 'Approved':
        return {'statusCode': 200, 'body': 'Not approved'}
    
    pkg = sagemaker.describe_model_package(ModelPackageName=model_pkg_arn)
    
    model_name = f"mlops-model-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    sagemaker.create_model(
        ModelName=model_name,
        PrimaryContainer=pkg['InferenceSpecification']['Containers'][0],
        ExecutionRoleArn=os.environ['SAGEMAKER_ROLE']
    )
    
    config_name = f"{model_name}-config"
    sagemaker.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            'VariantName': 'AllTraffic',
            'ModelName': model_name,
            'InitialInstanceCount': 1,
            'InstanceType': 'ml.t2.medium'
        }]
    )
    
    endpoint_name = 'mlops-production-endpoint'
    try:
        sagemaker.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
    except:
        sagemaker.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
    
    sns.publish(
        TopicArn=os.environ['SNS_TOPIC'],
        Subject='MLOps: Model Deployed',
        Message=f"Model deployed to {endpoint_name}\\n\\nModel: {model_pkg_arn}"
    )
    
    print(f"Deployed: {endpoint_name}")
    return {'statusCode': 200}
'''

with open('/tmp/lambda3.py', 'w') as f:
    f.write(lambda3_code)

with zipfile.ZipFile('/tmp/lambda3.zip', 'w') as z:
    z.write('/tmp/lambda3.py', 'lambda_function.py')

with open('/tmp/lambda3.zip', 'rb') as f:
    try:
        lambda3 = lambda_client.create_function(
            FunctionName='MLOps-DeployModel',
            Runtime='python3.11',
            Role=config['lambda_role_arn'],
            Handler='lambda_function.lambda_handler',
            Code={'ZipFile': f.read()},
            Timeout=900,
            Environment={'Variables': {
                'SAGEMAKER_ROLE': config['sagemaker_role_arn'],
                'SNS_TOPIC': config['sns_topic_arn']
            }}
        )
        lambda3_arn = lambda3['FunctionArn']
        print(f"✅ Lambda 3 created: {lambda3_arn}")
    except lambda_client.exceptions.ResourceConflictException:
        lambda3_arn = f"arn:aws:lambda:{config['region']}:{config['account_id']}:function:MLOps-DeployModel"
        print(f"✅ Lambda 3 already exists: {lambda3_arn}")

# Save Lambda ARNs to config
config['lambda1_arn'] = lambda1_arn
config['lambda2_arn'] = lambda2_arn
config['lambda3_arn'] = lambda3_arn

with open('mlops_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("\n✅ All Lambda functions created!")
print("Next step: Run 'python mlops_eventbridge.py' to create EventBridge rules")
