"""
Complete AWS cleanup for ML Specialty study projects
Cleans up: Lambda pipeline, SageMaker pipeline, Feature Store, and all related resources
"""
import boto3
import time

print("🧹 COMPLETE AWS CLEANUP")
print("="*70)
print("This will remove ALL resources from Weeks 7-8")
print("="*70)

confirm = input("\n⚠️  Are you sure? Type 'yes' to continue: ")
if confirm.lower() != 'yes':
    print("Cancelled.")
    exit(0)

# Initialize clients
sm = boto3.client('sagemaker')
lambda_client = boto3.client('lambda')
events = boto3.client('events')
sns = boto3.client('sns')
s3 = boto3.client('s3')
iam = boto3.client('iam')

print("\n" + "="*70)
print("STEP 1: SAGEMAKER ENDPOINTS")
print("="*70)

# Delete endpoints
endpoints = ['mlops-production-endpoint']
for endpoint_name in endpoints:
    try:
        print(f"\n🔍 Checking endpoint: {endpoint_name}")
        sm.describe_endpoint(EndpointName=endpoint_name)
        print(f"   Deleting endpoint...")
        sm.delete_endpoint(EndpointName=endpoint_name)
        print(f"   ✅ Endpoint deleted")
    except sm.exceptions.ClientError as e:
        if 'Could not find' in str(e):
            print(f"   ⏭️  Endpoint not found (already deleted)")
        else:
            print(f"   ⚠️  Error: {e}")

print("\n" + "="*70)
print("STEP 2: SAGEMAKER PIPELINES")
print("="*70)

# Delete pipelines
pipelines = ['MLOpsPipeline']
for pipeline_name in pipelines:
    try:
        print(f"\n🔍 Checking pipeline: {pipeline_name}")
        sm.describe_pipeline(PipelineName=pipeline_name)
        print(f"   Deleting pipeline...")
        sm.delete_pipeline(PipelineName=pipeline_name)
        print(f"   ✅ Pipeline deleted")
    except Exception as e:
        if 'does not exist' in str(e) or 'ResourceNotFound' in str(e):
            print(f"   ⏭️  Pipeline not found (already deleted)")
        else:
            print(f"   ⚠️  Error: {e}")

print("\n" + "="*70)
print("STEP 3: FEATURE STORE")
print("="*70)

# Delete feature groups
feature_groups = ['users-feature-group', 'products-feature-group']
for fg_name in feature_groups:
    try:
        print(f"\n🔍 Checking feature group: {fg_name}")
        sm.describe_feature_group(FeatureGroupName=fg_name)
        print(f"   Deleting feature group...")
        sm.delete_feature_group(FeatureGroupName=fg_name)
        print(f"   ✅ Feature group deleted")
    except Exception as e:
        if 'ResourceNotFound' in str(e):
            print(f"   ⏭️  Feature group not found (already deleted)")
        else:
            print(f"   ⚠️  Error: {e}")

print("\n" + "="*70)
print("STEP 4: SAGEMAKER MODELS & CONFIGS")
print("="*70)

# Delete models
print("\n📦 Checking models...")
try:
    models = sm.list_models(NameContains='mlops', MaxResults=50)
    if models['Models']:
        for model in models['Models']:
            model_name = model['ModelName']
            print(f"   Deleting model: {model_name}")
            sm.delete_model(ModelName=model_name)
            print(f"   ✅ Deleted")
    else:
        print("   ⏭️  No models found")
except Exception as e:
    print(f"   ⚠️  Error: {e}")

# Delete endpoint configs
print("\n⚙️  Checking endpoint configs...")
try:
    configs = sm.list_endpoint_configs(NameContains='mlops', MaxResults=50)
    if configs['EndpointConfigs']:
        for config in configs['EndpointConfigs']:
            config_name = config['EndpointConfigName']
            print(f"   Deleting config: {config_name}")
            sm.delete_endpoint_config(EndpointConfigName=config_name)
            print(f"   ✅ Deleted")
    else:
        print("   ⏭️  No configs found")
except Exception as e:
    print(f"   ⚠️  Error: {e}")

print("\n" + "="*70)
print("STEP 5: LAMBDA FUNCTIONS")
print("="*70)

# Delete Lambda functions
lambda_functions = ['MLOps-TriggerTraining', 'MLOps-RegisterModel', 'MLOps-DeployModel']
for func_name in lambda_functions:
    try:
        print(f"\n🔍 Checking Lambda: {func_name}")
        lambda_client.get_function(FunctionName=func_name)
        print(f"   Deleting function...")
        lambda_client.delete_function(FunctionName=func_name)
        print(f"   ✅ Function deleted")
    except lambda_client.exceptions.ResourceNotFoundException:
        print(f"   ⏭️  Function not found (already deleted)")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")

print("\n" + "="*70)
print("STEP 6: EVENTBRIDGE RULES")
print("="*70)

# Delete EventBridge rules
rules = ['MLOps-TrainingTrigger', 'MLOps-ModelApprovalTrigger', 'MLOps-TrainingCompleteTrigger']
for rule_name in rules:
    try:
        print(f"\n🔍 Checking EventBridge rule: {rule_name}")
        
        # Remove targets first
        targets = events.list_targets_by_rule(Rule=rule_name)
        if targets['Targets']:
            target_ids = [t['Id'] for t in targets['Targets']]
            events.remove_targets(Rule=rule_name, Ids=target_ids)
            print(f"   ✅ Removed targets")
        
        # Delete rule
        events.delete_rule(Name=rule_name)
        print(f"   ✅ Rule deleted")
    except events.exceptions.ResourceNotFoundException:
        print(f"   ⏭️  Rule not found (already deleted)")
    except Exception as e:
        print(f"   ⚠️  Error: {e}")

print("\n" + "="*70)
print("STEP 7: SNS TOPICS")
print("="*70)

# Delete SNS topics
try:
    print("\n📧 Checking SNS topics...")
    topics = sns.list_topics()
    mlops_topics = [t for t in topics['Topics'] if 'mlops' in t['TopicArn'].lower()]
    
    if mlops_topics:
        for topic in mlops_topics:
            topic_arn = topic['TopicArn']
            print(f"   Deleting topic: {topic_arn.split(':')[-1]}")
            sns.delete_topic(TopicArn=topic_arn)
            print(f"   ✅ Deleted")
    else:
        print("   ⏭️  No MLOps topics found")
except Exception as e:
    print(f"   ⚠️  Error: {e}")

print("\n" + "="*70)
print("STEP 8: S3 CLEANUP")
print("="*70)

bucket_name = 'sagemaker-us-east-2-854757836160'
print(f"\n🪣 Bucket: {bucket_name}")

cleanup_s3 = input("\n⚠️  Delete S3 objects? This will remove training data, models, logs. (yes/no): ")

if cleanup_s3.lower() == 'yes':
    prefixes = [
        'pipeline-code/',
        'pipeline-output/',
        'feature-store/',
        'mlops-training/',
        'mlops-output/'
    ]
    
    for prefix in prefixes:
        try:
            print(f"\n   Cleaning prefix: {prefix}")
            paginator = s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
            
            delete_count = 0
            for page in pages:
                if 'Contents' in page:
                    objects = [{'Key': obj['Key']} for obj in page['Contents']]
                    if objects:
                        s3.delete_objects(Bucket=bucket_name, Delete={'Objects': objects})
                        delete_count += len(objects)
            
            if delete_count > 0:
                print(f"   ✅ Deleted {delete_count} objects")
            else:
                print(f"   ⏭️  No objects found")
        except Exception as e:
            print(f"   ⚠️  Error: {e}")
else:
    print("   ⏭️  S3 cleanup skipped")

print("\n" + "="*70)
print("STEP 9: MODEL REGISTRY")
print("="*70)

# Model registry packages
print("\n📚 Checking Model Registry...")
delete_models = input("⚠️  Delete model packages from registry? (yes/no): ")

if delete_models.lower() == 'yes':
    try:
        packages = sm.list_model_packages(
            ModelPackageGroupName='mlops-pipeline-models',
            MaxResults=100
        )
        
        if packages['ModelPackageSummaryList']:
            for pkg in packages['ModelPackageSummaryList']:
                pkg_arn = pkg['ModelPackageArn']
                version = pkg_arn.split('/')[-1]
                print(f"   Deleting model package version {version}")
                sm.delete_model_package(ModelPackageName=pkg_arn)
                print(f"   ✅ Deleted")
        else:
            print("   ⏭️  No model packages found")
    except Exception as e:
        if 'does not exist' in str(e):
            print("   ⏭️  Model package group not found")
        else:
            print(f"   ⚠️  Error: {e}")
else:
    print("   ⏭️  Model registry cleanup skipped")

print("\n" + "="*70)
print("STEP 10: IAM ROLES (OPTIONAL)")
print("="*70)

print("\n🔐 IAM Roles:")
print("   - MLOpsSageMakerRole")
print("   - MLOpsLambdaRole")

delete_roles = input("\n⚠️  Delete IAM roles? Only do this if you're done with all projects. (yes/no): ")

if delete_roles.lower() == 'yes':
    roles = ['MLOpsSageMakerRole', 'MLOpsLambdaRole']
    
    for role_name in roles:
        try:
            print(f"\n   Processing role: {role_name}")
            
            # Detach policies
            attached_policies = iam.list_attached_role_policies(RoleName=role_name)
            for policy in attached_policies['AttachedPolicies']:
                print(f"      Detaching policy: {policy['PolicyName']}")
                iam.detach_role_policy(RoleName=role_name, PolicyArn=policy['PolicyArn'])
            
            # Delete inline policies
            inline_policies = iam.list_role_policies(RoleName=role_name)
            for policy_name in inline_policies['PolicyNames']:
                print(f"      Deleting inline policy: {policy_name}")
                iam.delete_role_policy(RoleName=role_name, PolicyName=policy_name)
            
            # Delete role
            iam.delete_role(RoleName=role_name)
            print(f"   ✅ Role deleted")
        except iam.exceptions.NoSuchEntityException:
            print(f"   ⏭️  Role not found")
        except Exception as e:
            print(f"   ⚠️  Error: {e}")
else:
    print("   ⏭️  IAM roles kept (can reuse for future projects)")

print("\n" + "="*70)
print("✅ CLEANUP COMPLETE!")
print("="*70)

print("\n📊 Summary:")
print("   ✅ Endpoints deleted")
print("   ✅ Pipelines deleted")
print("   ✅ Feature groups deleted")
print("   ✅ Models & configs deleted")
print("   ✅ Lambda functions deleted")
print("   ✅ EventBridge rules deleted")
print("   ✅ SNS topics deleted")
if cleanup_s3.lower() == 'yes':
    print("   ✅ S3 objects deleted")
if delete_models.lower() == 'yes':
    print("   ✅ Model packages deleted")
if delete_roles.lower() == 'yes':
    print("   ✅ IAM roles deleted")

print("\n💰 No ongoing costs!")
print("\n💡 Resources that remain (safe to keep):")
print("   - S3 bucket (empty = no cost)")
print("   - Model package group (metadata only = no cost)")
if delete_roles.lower() != 'yes':
    print("   - IAM roles (no cost)")

print("\n🎉 All done! Your AWS bill should drop to ~$0/day")