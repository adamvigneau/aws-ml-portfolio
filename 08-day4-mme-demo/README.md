# Multi-Model Endpoint (MME) Demo

A hands-on demonstration of SageMaker Multi-Model Endpoints for hosting multiple models on a single endpoint to reduce costs.

## Overview

This project demonstrates:

- **Training multiple models**: 5 XGBoost variants with different hyperparameters
- **Multi-Model Endpoint**: Host all models on ONE endpoint
- **Dynamic invocation**: Use TargetModel parameter to select which model
- **Cost savings**: 80%+ reduction compared to separate endpoints

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                Multi-Model Endpoint                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   S3 Bucket (Model Repository)                              │
│   ┌─────────────────────────────────┐                       │
│   │ model_conservative.tar.gz       │                       │
│   │ model_balanced.tar.gz           │                       │
│   │ model_aggressive.tar.gz         │◄─── Models stored     │
│   │ model_deep.tar.gz               │     in S3             │
│   │ model_fast.tar.gz               │                       │
│   └─────────────────────────────────┘                       │
│                    │                                        │
│                    ▼                                        │
│   ┌─────────────────────────────────┐                       │
│   │     Inference Container         │                       │
│   │  ┌───────────┐ ┌───────────┐   │                       │
│   │  │ Model A   │ │ Model B   │   │◄─── Models loaded     │
│   │  │ (cached)  │ │ (cached)  │   │     on demand         │
│   │  └───────────┘ └───────────┘   │                       │
│   └─────────────────────────────────┘                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Model Variants

| Model | num_round | max_depth | eta | Description |
|-------|-----------|-----------|-----|-------------|
| model_conservative | 25 | 2 | 0.3 | Simple, fast, low risk of overfitting |
| model_balanced | 50 | 4 | 0.2 | Balanced performance |
| model_aggressive | 100 | 6 | 0.1 | Complex, potentially more accurate |
| model_deep | 50 | 8 | 0.2 | Deep trees, captures complex patterns |
| model_fast | 30 | 3 | 0.4 | Optimized for speed |

## Scripts

| Script | Purpose |
|--------|---------|
| `train_multiple_models.py` | Train 5 XGBoost model variants |
| `deploy_multi_model_endpoint.py` | Deploy all models to single MME |
| `test_mme.py` | Test invoking different models |
| `cleanup_mme.py` | Delete all resources |

## Setup

### Prerequisites

- AWS account with SageMaker access
- Python 3.8+
- AWS CLI configured with credentials

### Installation

```bash
pip install 'sagemaker>=2.200.0,<3.0' boto3 pandas numpy
```

### Configuration

Update the `role` variable in each script with your SageMaker execution role ARN:

```python
role = "arn:aws:iam::YOUR_ACCOUNT:role/YOUR_SAGEMAKER_ROLE"
```

## Running the Demo

```bash
# 1. Train 5 model variants (~15-20 min)
python train_multiple_models.py

# 2. Deploy Multi-Model Endpoint (~5 min)
python deploy_multi_model_endpoint.py

# 3. Test invoking different models
python test_mme.py

# 4. Cleanup when done (important!)
python cleanup_mme.py
```

## Key Concept: TargetModel Parameter

The magic of MME — specify which model at inference time:

```python
response = runtime_client.invoke_endpoint(
    EndpointName='my-mme-endpoint',
    TargetModel='model_conservative.tar.gz',  # Select model here!
    ContentType='text/csv',
    Body=payload
)
```

Same endpoint, different model each request.

## Model Loading Behavior

| Call Type | Latency | Why |
|-----------|---------|-----|
| Cold start | 1-5 seconds | Model loaded from S3 |
| Warm call | 50-100 ms | Model cached in memory |

Frequently used models stay cached. Least-recently-used models get evicted.

## Cost Comparison

```
┌─────────────────────────────────────────────────────────────┐
│  Deployment Strategy      │  Endpoints  │  Cost/hour       │
├─────────────────────────────────────────────────────────────┤
│  5 Separate Endpoints     │     5       │  5 × $0.10 = $0.50│
│  1 Multi-Model Endpoint   │     1       │  1 × $0.10 = $0.10│
├─────────────────────────────────────────────────────────────┤
│  💰 SAVINGS               │    -4       │  80% cheaper!     │
└─────────────────────────────────────────────────────────────┘
```

**At scale:**

| Models | Separate Endpoints | MME | Savings |
|--------|-------------------|-----|---------|
| 10 | $1.00/hr | $0.10/hr | 90% |
| 100 | $10.00/hr | $0.10-0.30/hr | 97% |
| 1000 | $100.00/hr | $0.30-1.00/hr | 99% |

## MME vs MCE

| Aspect | Multi-Model (MME) | Multi-Container (MCE) |
|--------|-------------------|----------------------|
| Models | Many (pick one) | Few (chain all) |
| Framework | Same | Different |
| Invocation | TargetModel param | Serial pipeline |
| Use case | Per-customer, A/B | Preprocessing chains |

## Common Use Cases

1. **Per-customer models**: Personalized recommendations
2. **A/B testing**: Multiple model variants in production
3. **Regional models**: Different models for US, EU, APAC
4. **Time-based models**: Daily retraining with instant rollback

## Key Exam Concepts

1. **TargetModel** parameter specifies which model to invoke
2. Models loaded **on-demand** (first call = cold start)
3. All models must use **same framework/container**
4. Cost scales with endpoint, NOT number of models
5. MCE is for **chaining** models, MME is for **selecting** models

## Costs

- **Training**: ~$0.10 per model (ml.m5.large, ~3 min each)
- **Endpoint**: ~$0.10/hour for ml.m5.large
- **⚠️ Always run cleanup script when done!**

## Technologies

- AWS SageMaker Multi-Model Endpoints
- XGBoost
- Python / Boto3

## License

MIT
