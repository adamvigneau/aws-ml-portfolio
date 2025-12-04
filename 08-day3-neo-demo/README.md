# SageMaker Neo Demo

A hands-on demonstration of SageMaker Neo for compiling and optimizing ML models for specific hardware targets.

## Overview

This project demonstrates:

- **Model Training**: Train an XGBoost classifier
- **Neo Compilation**: Compile for Linux x86_64 target
- **Deployment**: Deploy both original and Neo-compiled models
- **Benchmarking**: Compare inference latency

## Architecture

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  Trained Model  │ ───▶ │  Neo Compiler   │ ───▶ │ Optimized Model │
│  (model.tar.gz) │      │                 │      │ + Runtime       │
│                 │      │  Target:        │      │                 │
│  Framework:     │      │  - OS           │      │  Runs on:       │
│  - XGBoost      │      │  - Architecture │      │  - Target HW    │
│  - TensorFlow   │      │  - Accelerator  │      │  - No framework │
│  - PyTorch      │      │                 │      │    needed       │
└─────────────────┘      └─────────────────┘      └─────────────────┘
```

## Scripts

| Script | Purpose |
|--------|---------|
| `train_model_for_neo.py` | Train XGBoost model on sample data |
| `compile_with_neo.py` | Compile model with SageMaker Neo |
| `deploy_and_benchmark.py` | Deploy both models and compare latency |
| `cleanup_neo_demo.py` | Delete all resources |

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
# 1. Train model (~3-5 min)
python train_model_for_neo.py

# 2. Compile with Neo (~3-5 min)
python compile_with_neo.py

# 3. Deploy and benchmark (~10 min)
python deploy_and_benchmark.py

# 4. Cleanup when done (important!)
python cleanup_neo_demo.py
```

## Neo Target Platforms

| Category | Examples |
|----------|----------|
| Cloud instances | ml.m5, ml.c5, ml.p3, ml.inf1 |
| NVIDIA devices | Jetson Nano, TX1, TX2, Xavier |
| ARM devices | Raspberry Pi, ARM64 Linux |
| Mobile | Android, iOS |

## Expected Results

For small models like XGBoost, performance is similar (network-bound):

| Metric | Original | Neo-Compiled |
|--------|----------|--------------|
| Average Latency | ~65 ms | ~65 ms |

Neo shows larger gains (1.5-4x) for:
- Deep learning models (CNN, BERT)
- Edge devices (Jetson, Raspberry Pi)
- When compute dominates latency

## Key Exam Concepts

1. **Neo compiles for specific hardware** - models are target-specific
2. **Runtime bundled** - no framework needed on device
3. **Supports**: TensorFlow, PyTorch, MXNet, XGBoost, ONNX
4. **Edge Manager** - for managing fleets of edge devices
5. **Use cases**: IoT, autonomous vehicles, mobile apps, low-latency inference

## Costs

- **Neo compilation**: Free (pay only for compute time)
- **Endpoints**: ~$0.10/hour for ml.m5.large
- **⚠️ Always run cleanup script when done!**

## Technologies

- AWS SageMaker Neo
- AWS SageMaker Endpoints
- XGBoost
- Python / Boto3

## License

MIT
