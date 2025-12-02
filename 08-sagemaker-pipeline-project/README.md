# SageMaker Pipelines - Automated ML Workflow

A production-grade ML pipeline using AWS SageMaker Pipelines with automated training, evaluation, and conditional model registration.

## Architecture
```
Data → Preprocessing → Training → Evaluation → Quality Gate → Model Registry
```

**Pipeline Steps:**
1. **PreprocessData** - Data preparation and train/validation split
2. **TrainModel** - XGBoost model training
3. **EvaluateModel** - Model evaluation with real metrics
4. **CheckAccuracy** - Quality gate (conditional logic)
5. **RegisterModel** - Register to Model Registry (if quality gate passes)

## Features

- ✅ **Visual DAG** - See pipeline flow in SageMaker Console
- ✅ **Parameter Management** - Change instance types and thresholds without code changes
- ✅ **Quality Gates** - Conditional model registration based on accuracy
- ✅ **Real Model Evaluation** - Load and evaluate actual trained models
- ✅ **Step Caching** - Automatic caching to save time and cost
- ✅ **Model Registry Integration** - Centralized model versioning

## Prerequisites

- AWS Account with SageMaker access
- AWS CLI configured
- Python 3.8+
- SageMaker execution role

## Installation
```bash
# Clone repository
git clone <your-repo-url>
cd sagemaker-pipeline-project

# Install dependencies
pip install -r requirements.txt

# Configure AWS credentials
aws configure
```

## Usage

### 1. Create Pipeline
```bash
python3 create_pipeline.py
```

### 2. Execute Pipeline
```bash
python3 -c "
import boto3
sm_client = boto3.client('sagemaker')
response = sm_client.start_pipeline_execution(
    PipelineName='MLOpsPipeline',
    PipelineParameters=[
        {'Name': 'ProcessingInstanceType', 'Value': 'ml.t3.medium'},
        {'Name': 'TrainingInstanceType', 'Value': 'ml.m5.xlarge'},
        {'Name': 'AccuracyThreshold', 'Value': '0.75'}
    ]
)
print(f'Execution ARN: {response[\"PipelineExecutionArn\"]}')
"
```

### 3. Monitor Execution
```bash
python3 monitor_pipeline.py <execution-arn>
```

Or view in AWS Console: [SageMaker Pipelines Console](https://console.aws.amazon.com/sagemaker/home#/pipelines)

### 4. Approve Model (If Registered)
```bash
# List pending models
aws sagemaker list-model-packages \
    --model-package-group-name mlops-pipeline-models \
    --model-approval-status PendingManualApproval

# Approve model
aws sagemaker update-model-package \
    --model-package-arn <model-arn> \
    --model-approval-status Approved
```

## Pipeline Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ProcessingInstanceType` | Instance for preprocessing/evaluation | `ml.m5.xlarge` |
| `TrainingInstanceType` | Instance for training | `ml.m5.xlarge` |
| `AccuracyThreshold` | Minimum accuracy for registration | `0.75` |

## Project Structure
```
sagemaker-pipeline-project/
├── preprocess.py          # Data preprocessing script
├── evaluate.py            # Model evaluation script
├── create_pipeline.py     # Pipeline definition
├── monitor_pipeline.py    # Execution monitoring
├── cleanup_week8.py       # Resource cleanup
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Cost Optimization

**Estimated costs per execution:**
- Processing (ml.t3.medium × 10 min): ~$0.01
- Training (ml.m5.xlarge × 10 min): ~$0.03
- Evaluation (ml.t3.medium × 5 min): ~$0.005
- **Total: ~$0.045 per pipeline run**

**Cost-saving features:**
- Step caching (skip unchanged steps)
- Spot instances (70% savings for training)
- Parameterized instance types

## Cleanup
```bash
python3 cleanup_week8.py
```

This removes:
- Pipeline definition
- Models and endpoint configs
- S3 pipeline artifacts
- Model packages (optional)

## Comparison: SageMaker Pipelines vs Lambda

| Feature | Lambda Pipeline | SageMaker Pipeline |
|---------|----------------|-------------------|
| **Visualization** | None | DAG in Console |
| **Quality Gates** | Manual code | Built-in conditions |
| **Parameters** | Environment vars | First-class |
| **Retry Logic** | Manual | Automatic |
| **Best For** | Simple workflows | Complex ML pipelines |

See [Lambda pipeline project](../mlops-pipeline/) for comparison.

## Learning Outcomes

This project demonstrates:
- ✅ ML workflow orchestration
- ✅ Conditional logic in pipelines
- ✅ Model evaluation and quality gates
- ✅ Parameter management
- ✅ Production MLOps patterns

Built as part of AWS ML Specialty certification study (Week 8 Day 1).

## Troubleshooting

### Issue: "Account-level service limit"
**Solution:** Use `ml.t3.medium` instead of `ml.m5.xlarge` for processing

### Issue: "Model evaluation fails"
**Solution:** Ensure XGBoost container is used for evaluation (not SKLearn)

### Issue: "Model not registered"
**Solution:** Check if accuracy meets threshold. Lower threshold or improve model.

## Resources

- [SageMaker Pipelines Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html)
- [Pipeline Step Types](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html)
- [Model Registry](https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html)

## License

MIT License

## Author

Built during AWS ML Specialty certification preparation.