import boto3
import json
import time
from datetime import datetime

# Initialize clients
iam = boto3.client('iam')
sns = boto3.client('sns')
sagemaker_client = boto3.client('sagemaker')
s3 = boto3.client('s3')

account_id = boto3.client('sts').get_caller_identity()['Account']
region = boto3.session.Session().region_name or 'us-east-1'

print("🚀 Starting MLOps pipeline setup...")
print(f"Account: {account_id}")
print(f"Region: {region}")

# 1. Get or create SageMaker role
print("\n1. Setting up SageMaker execution role...")
sm_role_arn = None

try:
    # Try to find existing role
    roles = iam.list_roles()
    for role in roles['Roles']:
        if 'SageMaker' in role['RoleName'] and 'ExecutionRole' in role['RoleName']:
            sm_role_arn = role['Arn']
            print(f"✅ Found existing SageMaker role: {role['RoleName']}")
            break
except Exception as e:
    print(f"Error listing roles: {e}")

if not sm_role_arn:
    print("Creating new SageMaker role...")
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }
    
    role = iam.create_role(
        RoleName='MLOpsSageMakerRole',
        AssumeRolePolicyDocument=json.dumps(trust_policy)
    )
    sm_role_arn = role['Role']['Arn']
    
    iam.attach_role_policy(
        RoleName='MLOpsSageMakerRole',
        PolicyArn='arn:aws:iam::aws:policy/AmazonSageMakerFullAccess'
    )
    iam.attach_role_policy(
        RoleName='MLOpsSageMakerRole',
        PolicyArn='arn:aws:iam::aws:policy/AmazonS3FullAccess'
    )
    print(f"✅ Created SageMaker role")
    time.sleep(10)

# 2. Create Lambda execution role
print("\n2. Setting up Lambda execution role...")
lambda_trust = {
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "lambda.amazonaws.com"},
        "Action": "sts:AssumeRole"
    }]
}

try:
    lambda_role = iam.create_role(
        RoleName='MLOpsLambdaRole',
        AssumeRolePolicyDocument=json.dumps(lambda_trust)
    )
    lambda_role_arn = lambda_role['Role']['Arn']
    
    iam.attach_role_policy(
        RoleName='MLOpsLambdaRole',
        PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
    )
    
    lambda_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Action": ["sagemaker:*", "s3:*", "sns:*", "iam:PassRole"],
            "Resource": "*"
        }]
    }
    iam.put_role_policy(
        RoleName='MLOpsLambdaRole',
        PolicyName='SageMakerAccess',
        PolicyDocument=json.dumps(lambda_policy)
    )
    print(f"✅ Lambda role created")
    time.sleep(10)
    
except iam.exceptions.EntityAlreadyExistsException:
    lambda_role_arn = f"arn:aws:iam::{account_id}:role/MLOpsLambdaRole"
    print(f"✅ Using existing Lambda role")

# 3. Create SNS topic
print("\n3. Setting up SNS notifications...")
try:
    topic = sns.create_topic(Name='mlops-pipeline-alerts')
    topic_arn = topic['TopicArn']
    print(f"✅ SNS topic created: {topic_arn}")
except Exception as e:
    topics = sns.list_topics()
    topic_arn = None
    for t in topics['Topics']:
        if 'mlops-pipeline-alerts' in t['TopicArn']:
            topic_arn = t['TopicArn']
            break
    print(f"✅ Using existing SNS topic: {topic_arn}")

# Ask for email
print("\n📧 Enter your email for notifications:")
email = input("Email: ").strip()

if email and '@' in email:
    try:
        sns.subscribe(
            TopicArn=topic_arn,
            Protocol='email',
            Endpoint=email
        )
        print(f"✅ Subscription sent to {email}")
        print("⚠️  CHECK YOUR EMAIL and confirm the subscription!")
    except Exception as e:
        print(f"Note: {e}")

# 4. Create Model Package Group
print("\n4. Setting up Model Registry...")
try:
    sagemaker_client.create_model_package_group(
        ModelPackageGroupName='mlops-pipeline-models',
        ModelPackageGroupDescription='Automated MLOps pipeline models'
    )
    print("✅ Model package group created: mlops-pipeline-models")
except sagemaker_client.exceptions.ResourceInUse:
    print("✅ Model package group already exists: mlops-pipeline-models")

# 5. Create S3 buckets
print("\n5. Setting up S3 buckets...")
training_bucket = f'mlops-training-{account_id}'
output_bucket = f'mlops-output-{account_id}'

for bucket in [training_bucket, output_bucket]:
    try:
        if region == 'us-east-1':
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={'LocationConstraint': region}
            )
        print(f"✅ Created bucket: {bucket}")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        print(f"✅ Using existing bucket: {bucket}")
    except Exception as e:
        print(f"Note for {bucket}: {e}")

# Enable EventBridge on training bucket
try:
    s3.put_bucket_notification_configuration(
        Bucket=training_bucket,
        NotificationConfiguration={'EventBridgeConfiguration': {}}
    )
    print(f"✅ EventBridge enabled on {training_bucket}")
except Exception as e:
    print(f"Note: {e}")

# 6. Save configuration
print("\n6. Saving configuration...")
config = {
    'account_id': account_id,
    'region': region,
    'sagemaker_role_arn': sm_role_arn,
    'lambda_role_arn': lambda_role_arn,
    'sns_topic_arn': topic_arn,
    'training_bucket': training_bucket,
    'output_bucket': output_bucket,
    'model_package_group': 'mlops-pipeline-models'
}

with open('mlops_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("✅ Configuration saved to mlops_config.json")

# Print summary
print("\n" + "="*70)
print("✅ SETUP COMPLETE!")
print("="*70)
print(f"\nResources created:")
print(f"  SageMaker Role: {sm_role_arn}")
print(f"  Lambda Role: {lambda_role_arn}")
print(f"  SNS Topic: {topic_arn}")
print(f"  Training Bucket: s3://{training_bucket}/")
print(f"  Output Bucket: s3://{output_bucket}/")
print(f"  Model Registry: mlops-pipeline-models")
print("\n📧 IMPORTANT: Check your email and confirm the SNS subscription!")
print("\nNext step: Run 'python mlops_lambdas.py' to create Lambda functions")
print("="*70)