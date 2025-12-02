"""
Execute SageMaker Pipeline from local machine
"""
import boto3
import sagemaker
import time
import json

# Get pipeline
region = boto3.Session().region_name
sm_client = boto3.client('sagemaker', region_name=region)

pipeline_name = "MLOpsPipeline"

print(f"🚀 Starting pipeline: {pipeline_name}")
print(f"Region: {region}")

# Start execution
try:
    response = sm_client.start_pipeline_execution(
        PipelineName=pipeline_name,
        PipelineParameters=[
            {
                'Name': 'AccuracyThreshold',
                'Value': '0.75'
            }
        ]
    )
    
    execution_arn = response['PipelineExecutionArn']
    execution_name = execution_arn.split('/')[-1]
    
    print(f"\n✅ Pipeline execution started!")
    print(f"Execution ARN: {execution_arn}")
    print(f"Execution name: {execution_name}")
    
    # Console link
    console_url = f"https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/pipelines/{pipeline_name}/executions/{execution_name}"
    print(f"\n📊 View in console:")
    print(console_url)
    
    # Monitor execution
    print(f"\n⏳ Pipeline is running (will take ~15-20 minutes)")
    print("Monitoring status every 60 seconds...")
    print("(Press Ctrl+C to stop monitoring, pipeline will continue running)\n")
    
    try:
        while True:
            # Get status
            desc = sm_client.describe_pipeline_execution(
                PipelineExecutionArn=execution_arn
            )
            
            status = desc['PipelineExecutionStatus']
            
            # Get step details
            steps = sm_client.list_pipeline_execution_steps(
                PipelineExecutionArn=execution_arn
            )
            
            # Print status
            print(f"\r[{time.strftime('%H:%M:%S')}] Pipeline: {status}  ", end='')
            
            # Show step progress
            step_summary = []
            for step in steps['PipelineExecutionSteps']:
                step_status = step['StepStatus']
                if step_status == 'Executing':
                    emoji = "⏳"
                elif step_status == 'Succeeded':
                    emoji = "✅"
                elif step_status == 'Failed':
                    emoji = "❌"
                else:
                    emoji = "⏸️"
                step_summary.append(f"{emoji} {step['StepName']}")
            
            print(f" | {' '.join(step_summary[:3])}", end='', flush=True)
            
            # Check if complete
            if status in ['Succeeded', 'Failed', 'Stopped']:
                print(f"\n\n{'='*70}")
                print(f"Pipeline {status}!")
                print(f"{'='*70}")
                
                # Show detailed results
                print("\nStep Results:")
                for step in steps['PipelineExecutionSteps']:
                    print(f"  {step['StepName']}: {step['StepStatus']}")
                    
                    if step['StepStatus'] == 'Failed' and 'FailureReason' in step:
                        print(f"    Error: {step['FailureReason']}")
                
                # Try to get metrics
                print("\nFetching model metrics...")
                for step in steps['PipelineExecutionSteps']:
                    if step['StepName'] == 'EvaluateModel' and step['StepStatus'] == 'Succeeded':
                        if 'Metadata' in step and 'ProcessingJob' in step['Metadata']:
                            job_name = step['Metadata']['ProcessingJob']['Arn'].split('/')[-1]
                            
                            try:
                                job_desc = sm_client.describe_processing_job(ProcessingJobName=job_name)
                                s3_uri = job_desc['ProcessingOutputConfig']['Outputs'][0]['S3Output']['S3Uri']
                                
                                # Download metrics
                                s3 = boto3.client('s3')
                                bucket = s3_uri.split('/')[2]
                                key = '/'.join(s3_uri.split('/')[3:]) + '/evaluation.json'
                                
                                obj = s3.get_object(Bucket=bucket, Key=key)
                                metrics = json.loads(obj['Body'].read().decode('utf-8'))
                                
                                print("\n📊 Model Metrics:")
                                for k, v in metrics.items():
                                    print(f"  {k}: {v:.4f}")
                            except Exception as e:
                                print(f"  Could not fetch metrics: {e}")
                
                break
            
            time.sleep(60)
            
    except KeyboardInterrupt:
        print(f"\n\n⏸️  Monitoring stopped (pipeline still running)")
        print(f"Check status at: {console_url}")
        
except Exception as e:
    print(f"\n❌ Error: {e}")