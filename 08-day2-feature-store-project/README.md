# SageMaker Feature Store Demo

A hands-on demonstration of AWS SageMaker Feature Store for managing ML features with online (real-time inference) and offline (training) stores.

## Overview

This project demonstrates key Feature Store concepts:

- **Feature Groups**: Schema definitions for related features
- **Online Store**: Low-latency feature retrieval for real-time inference (~60ms)
- **Offline Store**: Historical feature storage for training data (Athena/S3)
- **Point-in-Time Correctness**: Preventing data leakage in ML training

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SageMaker Feature Store                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐              ┌─────────────────┐          │
│  │  Feature Groups │              │  Feature Groups │          │
│  │  (users)        │              │  (products)     │          │
│  └────────┬────────┘              └────────┬────────┘          │
│           │                                │                    │
│           ▼                                ▼                    │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              Online Store (DynamoDB)                 │       │
│  │              - Real-time inference                   │       │
│  │              - Latest feature values                 │       │
│  │              - Single-digit ms latency               │       │
│  └─────────────────────────────────────────────────────┘       │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              Offline Store (S3 + Athena)             │       │
│  │              - Training data                         │       │
│  │              - Historical versions                   │       │
│  │              - Point-in-time queries                 │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Feature Groups

### Users Feature Group

| Feature | Type | Description |
|---------|------|-------------|
| user_id | String | Primary identifier |
| age | Integral | User age |
| membership_tier | String | bronze/silver/gold/platinum |
| total_purchases | Integral | Lifetime purchase count |
| avg_order_value | Fractional | Average order amount |
| event_time | Fractional | Timestamp for point-in-time |

### Products Feature Group

| Feature | Type | Description |
|---------|------|-------------|
| product_id | String | Primary identifier |
| category | String | Product category |
| price | Fractional | Current price |
| avg_rating | Fractional | Average customer rating |
| stock_level | Integral | Current inventory |
| event_time | Fractional | Timestamp for point-in-time |

## Scripts

| Script | Purpose |
|--------|---------|
| `create_feature_groups.py` | Create feature groups with online/offline stores |
| `ingest_features.py` | Ingest sample data into feature groups |
| `query_online_store.py` | Real-time feature queries for inference |
| `query_offline_store.py` | Athena queries for training data |
| `point_in_time_demo.py` | Demonstrate point-in-time correctness |
| `cleanup_feature_store.py` | Delete feature groups and resources |

## Setup

### Prerequisites

- AWS account with SageMaker access
- Python 3.8+
- AWS CLI configured with credentials

### Installation

```bash
# Create virtual environment
python3 -m venv fs-venv
source fs-venv/bin/activate

# Install dependencies
pip install 'sagemaker>=2.200.0,<3.0' boto3 pandas
```

### Running the Demo

```bash
# 1. Create feature groups (~2 min)
python create_feature_groups.py

# 2. Ingest sample data
python ingest_features.py

# 3. Query online store (real-time)
python query_online_store.py

# 4. Query offline store (wait ~15 min after ingestion)
python query_offline_store.py

# 5. Point-in-time demo
python point_in_time_demo.py

# 6. Cleanup when done
python cleanup_feature_store.py
```

## Key Concepts

### Online vs Offline Store

| Aspect | Online Store | Offline Store |
|--------|--------------|---------------|
| Use Case | Real-time inference | Training data |
| Latency | Single-digit ms | Seconds (Athena query) |
| Data | Latest values only | Full history |
| Backend | DynamoDB | S3 + Glue + Athena |

### Point-in-Time Correctness

Prevents data leakage by ensuring training features reflect what was known at each event time:

```sql
-- Point-in-time join: get features AS THEY WERE at transaction time
SELECT u.user_id, u.total_purchases, t.purchase_amount
FROM users_feature_group u
JOIN transactions t
  ON u.user_id = t.user_id
  AND u.event_time <= t.event_time  -- Key: only use past features
```

**Why it matters:**
- Training with "future" features = data leakage
- Model performs great in training, fails in production
- Feature Store solves this with `event_time` tracking

## AWS Exam Tips

Key concepts for AWS ML Specialty certification:

1. **Feature Store = Single source of truth** for both training and inference
2. **Online store** for real-time, **offline store** for batch/training
3. **Point-in-time correctness** prevents data leakage
4. **event_time** column enables historical feature lookups
5. Feature groups support **both stores simultaneously**

## Costs

- **Online Store**: ~$1.25/GB-month (DynamoDB)
- **Offline Store**: S3 storage + Athena queries
- **Tip**: Delete endpoints/feature groups when not in use

## Technologies

- AWS SageMaker Feature Store
- Amazon DynamoDB (online store backend)
- Amazon S3 (offline store)
- AWS Glue (data catalog)
- Amazon Athena (SQL queries)
- Python / Boto3

## License

MIT
