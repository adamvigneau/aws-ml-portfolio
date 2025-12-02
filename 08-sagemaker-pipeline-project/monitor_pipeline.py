"""
Monitor pipeline execution
"""
import boto3
import time
import sys
from datetime import datetime

def monitor_execution(execution_arn):
    """
    Monitor pipeline execution with real-time updates
    """
    
    sm_client = boto3.client('sagemaker')
    
    pipeline_name = execution_arn.split('/')[-3]
    exec_name = execution_arn.split('/')[-1]
    
    print(f"📊 Monitoring Pipeline Execution")
    print("="*70)
    print(f"Pipeline: {pipeline_name}")
    print(f"Execution: {exec_name}")
    print("="*70)
    
    last_status = None
    
    while True:
        try:
            # Get execution details
            desc = sm_client.describe_pipeline_execution(
                PipelineExecutionArn=execution_arn
            )
            
            status = desc['PipelineExecutionStatus']
            
            # List steps
            steps_response = sm_client.list_pipeline_execution_steps(
                PipelineExecutionArn=execution_arn
            )
            
            # Only print if status changed
            if status != last_status:
                print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - Status: {status}")
                print("-"*70)
                
                for step in steps_response['PipelineExecutionSteps']:
                    step_name = step['StepName']
                    step_status = step['StepStatus']
                    
                    # Status emoji
                    if step_status == 'Executing':
                        emoji = "⏳"
                    elif step_status == 'Succeeded':
                        emoji = "✅"
                    elif step_status == 'Failed':
                        emoji = "❌"
                    elif step_status == 'Stopping':
                        emoji = "⏸️"
                    else:
                        emoji = "⚪"
                    
                    print(f"{emoji} {step_name}: {step_status}")
                    
                    # Show duration if completed
                    if 'StartTime' in step and 'EndTime' in step:
                        duration = (step['EndTime'] - step['StartTime']).total_seconds()
                        print(f"     Duration: {duration:.0f}s")
                    
                    # Show failure reason
                    if step_status == 'Failed' and 'FailureReason' in step:
                        print(f"     ❌ {step['FailureReason']}")
                
                last_status = status
            
            # Break if completed
            if status in ['Succeeded', 'Failed', 'Stopped']:
                print("\n" + "="*70)
                if status == 'Succeeded':
                    print("✅ Pipeline execution SUCCEEDED!")
                elif status == 'Failed':
                    print("❌ Pipeline execution FAILED!")
                else:
                    print("⏸️  Pipeline execution STOPPED!")
                print("="*70)
                break
            
            # Wait before next check
            time.sleep(30)
            
        except KeyboardInterrupt:
            print("\n\n⏸️  Monitoring stopped (pipeline still running)")
            print(f"Resume with: python3 monitor_pipeline.py {execution_arn}")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            time.sleep(30)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 monitor_pipeline.py <execution_arn>")
        sys.exit(1)
    
    execution_arn = sys.argv[1]
    monitor_execution(execution_arn)