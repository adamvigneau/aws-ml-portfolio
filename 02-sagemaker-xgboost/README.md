## Week 2 Project: Titanic Prediction with SageMaker XGBoost

### What I Built
- End-to-end ML pipeline on AWS SageMaker
- Used XGBoost built-in algorithm for binary classification
- Deployed real-time prediction endpoint

### Architecture
1. Data preprocessing locally
2. Upload to S3
3. SageMaker training job (ml.m5.xlarge)
4. Model deployment to endpoint
5. Real-time predictions via REST API

### Results
- Training AUC: ~0.85-0.90
- Predictions working correctly
- Successfully deployed and cleaned up resources

### Key Learnings
- SageMaker workflow (data → S3 → train → deploy)
- XGBoost hyperparameter tuning
- Endpoint management and cost control
- Difference between local Jupyter and cloud training