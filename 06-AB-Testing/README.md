# A/B Testing with Amazon SageMaker Multi-Variant Endpoints

## Project Overview

This project demonstrates production-grade A/B testing for machine learning models using Amazon SageMaker's multi-variant endpoint feature. It showcases how to deploy two model variants with different hyperparameters, split traffic between them, and monitor performance metrics to make data-driven decisions about model deployment.

**Business Use Case:** Organizations often need to test new model versions in production without fully replacing existing models. This gradual rollout approach (canary deployment) minimizes risk while gathering real-world performance data.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SageMaker Endpoint                        │
│  ┌────────────────────────────┬──────────────────────────┐  │
│  │   VariantA (Conservative)  │   VariantB (Aggressive)  │  │
│  │   • Weight: 80%            │   • Weight: 20%          │  │
│  │   • num_round=50           │   • num_round=100        │  │
│  │   • max_depth=3            │   • max_depth=6          │  │
│  │   • eta=0.3                │   • eta=0.1              │  │
│  └────────────────────────────┴──────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │ Traffic Distribution
                            │
                   ┌────────┴────────┐
                   │   API Gateway   │
                   │  or Application │
                   └─────────────────┘
```

---

## Key Features

✅ **Multi-variant deployment** with traffic splitting (80/20)  
✅ **Two model variants** with different hyperparameters  
✅ **CloudWatch monitoring** for invocations and latency  
✅ **Canary deployment pattern** for safe rollouts  
✅ **Production-ready** A/B testing infrastructure  
✅ **Cost optimization** with proper resource cleanup  

---

## Technologies Used

- **AWS SageMaker** - Model training and deployment
- **XGBoost** - Gradient boosting algorithm
- **CloudWatch** - Metrics and monitoring
- **Boto3** - AWS SDK for Python
- **Python 3.x** - Programming language

---

## Dataset

**Titanic Dataset** - Binary classification problem predicting passenger survival

**Features:**
- `Pclass` - Passenger class (1, 2, 3)
- `Sex` - Gender (0=male, 1=female)
- `Age` - Age in years
- `SibSp` - Number of siblings/spouses aboard
- `Parch` - Number of parents/children aboard
- `Fare` - Ticket fare
- `Embarked` - Port of embarkation (0=S, 1=C, 2=Q)

**Target:** `Survived` (0=No, 1=Yes)

---

## Model Variants

### Variant A: Conservative (Baseline)
```python
Hyperparameters:
  - num_round: 50         # Fewer training rounds
  - max_depth: 3          # Shallower trees
  - eta: 0.3              # Higher learning rate
  - subsample: 1.0        # Use all data
  
Performance:
  - Average Latency: 5.3 seconds
  - Traffic Weight: 80%
```

### Variant B: Aggressive (Challenger)
```python
Hyperparameters:
  - num_round: 100        # More training rounds
  - max_depth: 6          # Deeper trees
  - eta: 0.1              # Lower learning rate
  - subsample: 0.8        # Use 80% of data
  
Performance:
  - Average Latency: 8.6 seconds (63% slower)
  - Traffic Weight: 20%
```

---

## Project Structure

```
ab-testing-sagemaker/
├── phase1_train_models.py      # Train both model variants
├── phase2_deploy_endpoint.py   # Deploy multi-variant endpoint
├── phase3_test_traffic.py      # Verify traffic splitting
├── phase4_monitor_metrics.py   # CloudWatch monitoring
├── phase5_cleanup.py           # Resource cleanup
└── README.md                   # This file
```

---

## Implementation Guide

### Phase 1: Train Two Model Variants (1 hour)

**Objective:** Train two XGBoost models with different hyperparameters

```python
import sagemaker
from sagemaker.estimator import Estimator

# Model A: Conservative
xgb_model_a = Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type='ml.m5.xlarge',
    output_path=f's3://{bucket}/ab-test/model-a/'
)
xgb_model_a.set_hyperparameters(
    objective='binary:logistic',
    num_round=50,
    max_depth=3,
    eta=0.3
)
xgb_model_a.fit({'train': train_path})

# Model B: Aggressive (similar setup, different hyperparameters)
```

**Output:** Two trained models saved to S3

---

### Phase 2: Deploy Multi-Variant Endpoint (15 minutes)

**Objective:** Create endpoint with both variants and 80/20 traffic split

```python
# Register both models
client.create_model(
    ModelName=model_a_name,
    PrimaryContainer={'Image': container, 'ModelDataUrl': model_a_data},
    ExecutionRoleArn=role
)

# Create endpoint configuration with traffic split
client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            'VariantName': 'VariantA',
            'ModelName': model_a_name,
            'InitialInstanceCount': 1,
            'InstanceType': 'ml.t2.medium',
            'InitialVariantWeight': 80  # 80% of traffic
        },
        {
            'VariantName': 'VariantB',
            'ModelName': model_b_name,
            'InitialInstanceCount': 1,
            'InstanceType': 'ml.t2.medium',
            'InitialVariantWeight': 20  # 20% of traffic
        }
    ]
)

# Deploy endpoint
client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)
```

**Output:** Live endpoint with two variants receiving traffic

---

### Phase 3: Test Traffic Split (5 minutes)

**Objective:** Verify that traffic is distributed according to weights

```python
# Send 100 test predictions
variant_counts = []
runtime_client = boto3.client('sagemaker-runtime')

for i in range(100):
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=test_sample,
        ContentType='text/csv'
    )
    variant_name = response['InvokedProductionVariant']
    variant_counts.append(variant_name)

# Analyze distribution
from collections import Counter
distribution = Counter(variant_counts)
print(f"VariantA: {distribution['VariantA']}%")
print(f"VariantB: {distribution['VariantB']}%")
```

**Expected Results:**
- VariantA: ~80% of requests (±10%)
- VariantB: ~20% of requests (±10%)

**Actual Results:**
- VariantA: 86% (✅)
- VariantB: 14% (✅)

---

### Phase 4: Monitor with CloudWatch (10 minutes)

**Objective:** Track performance metrics for both variants

```python
cloudwatch = boto3.client('cloudwatch')

# Get invocation metrics
response = cloudwatch.get_metric_statistics(
    Namespace='AWS/SageMaker',
    MetricName='Invocations',
    Dimensions=[
        {'Name': 'EndpointName', 'Value': endpoint_name},
        {'Name': 'VariantName', 'Value': 'VariantA'}
    ],
    StartTime=start_time,
    EndTime=end_time,
    Period=300,
    Statistics=['Sum']
)

# Get latency metrics
response = cloudwatch.get_metric_statistics(
    Namespace='AWS/SageMaker',
    MetricName='ModelLatency',
    # ... similar setup
    Statistics=['Average', 'Maximum', 'Minimum']
)
```

**Key Metrics Tracked:**
- Invocations per variant
- Average/min/max latency
- Error rates (if any)
- Traffic distribution

---

### Phase 5: Cleanup (2 minutes)

**Objective:** Delete all resources to avoid charges

```python
# Delete endpoint (CRITICAL - stops billing)
client.delete_endpoint(EndpointName=endpoint_name)

# Delete endpoint config
client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)

# Delete models (optional)
client.delete_model(ModelName=model_a_name)
client.delete_model(ModelName=model_b_name)
```

---

## Results & Insights

### Traffic Distribution
| Variant | Expected | Actual | Status |
|---------|----------|--------|--------|
| VariantA | 80% | 84.9% | ✅ Working |
| VariantB | 20% | 15.1% | ✅ Working |

### Performance Comparison
| Metric | VariantA (Conservative) | VariantB (Aggressive) | Winner |
|--------|-------------------------|------------------------|--------|
| **Avg Latency** | 5.3 seconds | 8.6 seconds | VariantA |
| **Min Latency** | 3.8 seconds | 4.7 seconds | VariantA |
| **Max Latency** | 23.2 seconds | 94.9 seconds | VariantA |
| **Traffic Handled** | 129 requests | 23 requests | VariantA |

### Key Finding
**VariantB is 63% slower than VariantA** despite more aggressive hyperparameters aimed at accuracy. This demonstrates the classic **accuracy vs. latency tradeoff** in production ML systems.

**Production Decision:** In a real scenario, you would:
1. Run the A/B test for 1-2 weeks
2. Compare actual prediction accuracy on production data
3. Measure business impact (conversions, revenue, etc.)
4. Decide if accuracy gains justify the latency cost
5. Gradually shift traffic if winner is clear (e.g., 90/10, then 100/0)

---

## Cost Analysis

### Active Deployment Costs
| Resource | Instance Type | Cost/Hour | Quantity | Total/Hour |
|----------|--------------|-----------|----------|------------|
| VariantA | ml.t2.medium | $0.065 | 1 | $0.065 |
| VariantB | ml.t2.medium | $0.065 | 1 | $0.065 |
| **Total** | | | | **$0.13/hour** |

### Testing Duration Costs
- **1 week test:** $0.13 × 168 hours = **$21.84**
- **After cleanup:** **$0/hour** ✅

### S3 Storage (Post-Cleanup)
- Training data: ~5 MB
- Model artifacts: ~10 MB each
- **Total cost:** ~$0.01/month (negligible)

---

## Production Best Practices Demonstrated

### 1. Canary Deployment Pattern
- Start with small percentage (20%) to limit risk
- Monitor closely before increasing traffic
- Gradual rollout strategy (20% → 50% → 100%)

### 2. Traffic Management
```python
# Initial deployment
VariantA: 80%, VariantB: 20%

# If VariantB performs better, shift gradually:
Week 2: VariantA: 50%, VariantB: 50%
Week 3: VariantA: 20%, VariantB: 80%
Week 4: VariantA: 0%, VariantB: 100%
```

### 3. Monitoring & Observability
- CloudWatch for real-time metrics
- Track invocations, latency, errors
- Set up alarms for anomalies
- Compare business KPIs

### 4. Rollback Strategy
```python
# If VariantB fails, immediately shift traffic back
client.update_endpoint_weights_and_capacities(
    EndpointName=endpoint_name,
    DesiredWeightsAndCapacities=[
        {'VariantName': 'VariantA', 'DesiredWeight': 100},
        {'VariantName': 'VariantB', 'DesiredWeight': 0}
    ]
)
```

### 5. Cost Optimization
- Use smaller instance types for testing (ml.t2.medium)
- Delete endpoints when not in use
- Keep model artifacts in S3 for quick redeployment

---

## Common Issues & Solutions

### Issue 1: ResourceLimitExceeded
**Problem:** Account quota limits for instance types

**Solution:**
```python
# Use smaller instances that fit within quota
instance_type='ml.t2.medium'  # Instead of ml.m5.xlarge
```

### Issue 2: Model Not Found Error
**Problem:** Model not registered before endpoint config creation

**Solution:**
```python
# Always call create_model() before using in endpoint config
client.create_model(ModelName=model_name, ...)
time.sleep(2)  # Brief pause to ensure registration
client.create_endpoint_config(...)
```

### Issue 3: CloudWatch Metrics Delayed
**Problem:** Metrics take 5-15 minutes to appear

**Solution:**
```python
# Wait after generating traffic
time.sleep(120)  # 2-minute delay
# Then query CloudWatch
```

### Issue 4: Endpoint Already Exists
**Problem:** Trying to create endpoint with existing name

**Solution:**
```python
# Always check and delete old endpoints first
try:
    client.delete_endpoint(EndpointName=endpoint_name)
    time.sleep(30)  # Wait for deletion
except:
    pass
```

---

## Extensions & Next Steps

### 1. Add Model Quality Metrics
```python
# Track prediction accuracy in production
def log_prediction_quality(prediction, actual):
    cloudwatch.put_metric_data(
        Namespace='ML/ModelQuality',
        MetricData=[{
            'MetricName': 'PredictionAccuracy',
            'Value': 1.0 if prediction == actual else 0.0,
            'Dimensions': [{'Name': 'VariantName', 'Value': variant}]
        }]
    )
```

### 2. Automated Traffic Shifting
```python
# Automatically promote winner after 1 week
if variant_b_accuracy > variant_a_accuracy + 0.02:  # 2% improvement
    shift_traffic(variant_b, percentage=100)
```

### 3. Multi-Armed Bandit
```python
# Dynamic traffic allocation based on performance
# Winners automatically get more traffic
```

### 4. Model Registry Integration
```python
# Track all model versions in SageMaker Model Registry
# Enable model lineage and governance
```

### 5. CI/CD Pipeline
```python
# Automate: Train → Test → Deploy → Monitor
# Use SageMaker Pipelines or AWS Step Functions
```

---

## Learning Outcomes

By completing this project, you've learned:

✅ **Multi-variant endpoint deployment** - Core SageMaker production feature  
✅ **Traffic splitting strategies** - Canary and blue-green deployments  
✅ **Production monitoring** - CloudWatch metrics and alarms  
✅ **Model comparison methodology** - Data-driven model selection  
✅ **Cost management** - AWS resource optimization  
✅ **Risk mitigation** - Safe model rollout patterns  
✅ **Latency vs. accuracy tradeoffs** - Production ML considerations  

---

## AWS Certification Relevance

This project covers key topics for:

### AWS Certified Machine Learning - Specialty
- **Domain 3: Modeling** (20% of exam)
  - Model performance optimization
  - Hyperparameter tuning impact
  
- **Domain 4: Machine Learning Implementation and Operations** (36% of exam)
  - Multi-variant endpoint deployment ⭐
  - A/B testing strategies ⭐
  - Production monitoring
  - Model versioning
  - Canary deployments ⭐

### Exam-Style Questions You Can Now Answer

**Q:** How would you safely deploy a new model version to production?  
**A:** Use SageMaker multi-variant endpoints with traffic splitting. Start with 10-20% traffic to the new variant, monitor metrics, and gradually increase if performance is better.

**Q:** What metrics should you monitor for production ML endpoints?  
**A:** Invocations, latency (avg/min/max), error rates, model drift, and business KPIs specific to the use case.

**Q:** How do you compare two model versions in production?  
**A:** Deploy both as variants on a single endpoint, split traffic (e.g., 80/20), track CloudWatch metrics, and compare performance over time before fully promoting the winner.

---

## References

- [SageMaker Multi-Variant Endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/model-ab-testing.html)
- [SageMaker Production Variants](https://docs.aws.amazon.com/sagemaker/latest/dg/deployment-best-practices.html)
- [CloudWatch Metrics for SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/monitoring-cloudwatch.html)
- [XGBoost Hyperparameters](https://xgboost.readthedocs.io/en/latest/parameter.html)

---

## Contact & Questions

This project was completed as part of AWS ML certification preparation.

**Timeline:** 4-5 hours total
- Phase 1 (Training): 1 hour
- Phase 2 (Deployment): 15 minutes
- Phase 3 (Testing): 5 minutes
- Phase 4 (Monitoring): 10 minutes
- Phase 5 (Cleanup): 2 minutes
- Documentation: 1 hour

---

## License

This project is for educational purposes as part of AWS certification preparation.

---

**🎉 Project Complete!** You now have hands-on experience with production ML deployment patterns used by companies like Amazon, Netflix, and Uber for safely rolling out model improvements.