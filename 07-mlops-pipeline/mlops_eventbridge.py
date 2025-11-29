import boto3
import json

# Load config
with open('mlops_config.json', 'r') as f:
    config = json.load(f)

events = boto3.client('events')
lambda_client = boto3.client('lambda')

print("🚀 Creating EventBridge rules...")

# Rule 1: S3 upload → Trigger training
print("\n1. Creating rule: S3 Upload → Training...")

rule1 = events.put_rule(
    Name='MLOps-S3Upload',
    EventPattern=json.dumps({
        "source": ["aws.s3"],
        "detail-type": ["Object Created"],
        "detail": {
            "bucket": {"name": [config['training_bucket']]},
            "object": {"key": [{"prefix": "training-data/"}]}
        }
    }),
    State='ENABLED'
)

events.put_targets(
    Rule='MLOps-S3Upload',
    Targets=[{'Id': '1', 'Arn': config['lambda1_arn']}]
)

lambda_client.add_permission(
    FunctionName='MLOps-TriggerTraining',
    StatementId='EventBridgeInvoke',
    Action='lambda:InvokeFunction',
    Principal='events.amazonaws.com',
    SourceArn=rule1['RuleArn']
)

print(f"✅ Rule created: S3 → Training")

# Rule 2: Training complete → Register model
print("\n2. Creating rule: Training Complete → Registration...")

rule2 = events.put_rule(
    Name='MLOps-TrainingComplete',
    EventPattern=json.dumps({
        "source": ["aws.sagemaker"],
        "detail-type": ["SageMaker Training Job State Change"],
        "detail": {"TrainingJobStatus": ["Completed"]}
    }),
    State='ENABLED'
)

events.put_targets(
    Rule='MLOps-TrainingComplete',
    Targets=[{'Id': '1', 'Arn': config['lambda2_arn']}]
)

lambda_client.add_permission(
    FunctionName='MLOps-RegisterModel',
    StatementId='EventBridgeInvoke',
    Action='lambda:InvokeFunction',
    Principal='events.amazonaws.com',
    SourceArn=rule2['RuleArn']
)

print(f"✅ Rule created: Training → Registration")

# Rule 3: Model approved → Deploy
print("\n3. Creating rule: Model Approved → Deployment...")

rule3 = events.put_rule(
    Name='MLOps-ModelApproved',
    EventPattern=json.dumps({
        "source": ["aws.sagemaker"],
        "detail-type": ["SageMaker Model Package State Change"],
        "detail": {"ModelApprovalStatus": ["Approved"]}
    }),
    State='ENABLED'
)

events.put_targets(
    Rule='MLOps-ModelApproved',
    Targets=[{'Id': '1', 'Arn': config['lambda3_arn']}]
)

lambda_client.add_permission(
    FunctionName='MLOps-DeployModel',
    StatementId='EventBridgeInvoke',
    Action='lambda:InvokeFunction',
    Principal='events.amazonaws.com',
    SourceArn=rule3['RuleArn']
)

print(f"✅ Rule created: Approval → Deployment")

print("\n" + "="*70)
print("✅ PIPELINE COMPLETE!")
print("="*70)
print("\nReady to test! Run 'python mlops_test.py' to trigger the pipeline")
