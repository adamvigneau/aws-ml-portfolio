"""
Analyze data drift and demonstrate what Model Monitor detects
"""
import boto3
import json
import numpy as np

# Configuration
region = boto3.Session().region_name
s3_client = boto3.client('s3', region_name=region)

print(f"Region: {region}")

# Load config
with open('monitoring_config.json', 'r') as f:
    config = json.load(f)

bucket = config['bucket']
prefix = config['prefix']
endpoint_name = config['endpoint_name']

print("\n" + "="*70)
print("ANALYZING DATA DRIFT")
print("="*70)

# ============================================================
# Load Baseline Statistics
# ============================================================
print("\n" + "-"*70)
print("BASELINE STATISTICS (from training data)")
print("-"*70)

stats_key = f"{prefix}/baseline-results/statistics.json"
try:
    response = s3_client.get_object(Bucket=bucket, Key=stats_key)
    statistics = json.loads(response['Body'].read().decode('utf-8'))
    
    baseline_stats = {}
    
    print(f"\n{'Feature':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("-" * 70)
    
    for feature in statistics.get('features', []):
        name = feature.get('name', 'unknown')
        numerical = feature.get('numerical_statistics', {})
        
        mean = numerical.get('mean', 0)
        std = numerical.get('std', 0)
        min_val = numerical.get('min', 0)
        max_val = numerical.get('max', 0)
        
        baseline_stats[name] = {'mean': mean, 'std': std, 'min': min_val, 'max': max_val}
        
        # Handle different data types
        try:
            print(f"{name:<20} {float(mean):>12.2f} {float(std):>12.2f} {float(min_val):>12.2f} {float(max_val):>12.2f}")
        except (ValueError, TypeError):
            print(f"{name:<20} {str(mean):>12} {str(std):>12} {str(min_val):>12} {str(max_val):>12}")
            
except Exception as e:
    print(f"   Could not load baseline statistics: {e}")
    # Use expected values from training
    baseline_stats = {
        'age': {'mean': 35, 'std': 10},
        'income': {'mean': 50000, 'std': 15000},
        'credit_score': {'mean': 700, 'std': 50},
        'loan_amount': {'mean': 25000, 'std': 10000},
        'employment_years': {'mean': 8, 'std': 5}
    }
    print("\n   Using expected baseline values from training script")

# ============================================================
# Simulated Drifted Statistics
# ============================================================
print("\n" + "-"*70)
print("DRIFTED DATA STATISTICS (from inference traffic)")
print("-"*70)

# These are the distributions we used for drifted traffic
drifted_stats = {
    'age': {'mean': 45, 'std': 15},
    'income': {'mean': 35000, 'std': 10000},
    'credit_score': {'mean': 620, 'std': 80},
    'loan_amount': {'mean': 40000, 'std': 15000},
    'employment_years': {'mean': 3, 'std': 2}
}

print(f"\n{'Feature':<20} {'Mean':>12} {'Std':>12}")
print("-" * 46)

for name, stats in drifted_stats.items():
    print(f"{name:<20} {stats['mean']:>12.2f} {stats['std']:>12.2f}")

# ============================================================
# Drift Analysis
# ============================================================
print("\n" + "="*70)
print("DRIFT DETECTION ANALYSIS")
print("="*70)

print("""
Model Monitor uses statistical tests to detect drift:
- Kolmogorov-Smirnov test (distribution comparison)
- Population Stability Index (PSI)
- Simple threshold-based checks (mean shift > X%)
""")

print(f"\n{'Feature':<20} {'Baseline':>12} {'Drifted':>12} {'Change':>12} {'Alert':>10}")
print("-" * 70)

drift_detected = []

for feature in ['age', 'income', 'credit_score', 'loan_amount', 'employment_years']:
    baseline_mean = baseline_stats.get(feature, {}).get('mean', 0)
    drifted_mean = drifted_stats.get(feature, {}).get('mean', 0)
    
    try:
        baseline_mean = float(baseline_mean)
        change_pct = ((drifted_mean - baseline_mean) / baseline_mean) * 100
        
        # Alert if change > 20%
        alert = "🚨 DRIFT" if abs(change_pct) > 20 else "✅ OK"
        if abs(change_pct) > 20:
            drift_detected.append(feature)
        
        print(f"{feature:<20} {baseline_mean:>12.1f} {drifted_mean:>12.1f} {change_pct:>+11.1f}% {alert:>10}")
    except (ValueError, TypeError):
        print(f"{feature:<20} {'N/A':>12} {drifted_mean:>12.1f} {'N/A':>12} {'?':>10}")

# ============================================================
# Check for Captured Data
# ============================================================
print("\n" + "-"*70)
print("DATA CAPTURE STATUS")
print("-"*70)

capture_prefix = f"{prefix}/data-capture/{endpoint_name}/AllTraffic"
try:
    response = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix=capture_prefix,
        MaxKeys=20
    )
    
    files = response.get('Contents', [])
    if files:
        print(f"\n   ✅ Found {len(files)} captured data files!")
        print(f"   Location: s3://{bucket}/{capture_prefix}/")
        
        # Show files
        print("\n   Recent capture files:")
        for f in files[:5]:
            print(f"   - {f['Key'].split('/')[-1]}")
    else:
        print(f"\n   ⚠️  No captured data files yet")
        print(f"   Data capture can take up to 5 minutes to appear in S3")
        
except Exception as e:
    print(f"\n   ⚠️  Could not check data capture: {e}")

# ============================================================
# Summary
# ============================================================
print("\n" + "="*70)
print("DRIFT DETECTION SUMMARY")
print("="*70)

print(f"""
Features with detected drift (>20% change):
""")

for feature in drift_detected:
    print(f"   🚨 {feature}")

if not drift_detected:
    print("   ✅ No significant drift detected")

print(f"""
┌─────────────────────────────────────────────────────────────────┐
│  WHAT WOULD HAPPEN IN PRODUCTION                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Model Monitor scheduled job runs (hourly/daily)             │
│  2. Compares captured data against baseline                     │
│  3. Detects drift in {len(drift_detected)} features                               │
│  4. Generates violation report in S3                            │
│  5. Triggers CloudWatch alarm                                   │
│  6. Team investigates and decides:                              │
│     - Retrain model with new data?                              │
│     - Roll back to previous model?                              │
│     - Adjust business rules?                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
""")

print("\n" + "-"*70)
print("KEY EXAM CONCEPTS")
print("-"*70)
print("""
1. DATA DRIFT: Input feature distributions change over time
   → Detected by comparing against baseline statistics

2. CONCEPT DRIFT: Relationship between features and target changes
   → Detected by monitoring model quality metrics (accuracy, etc.)

3. MODEL MONITOR TYPES:
   - Data Quality Monitor: Detects data drift
   - Model Quality Monitor: Detects accuracy degradation
   - Bias Drift Monitor: Detects fairness metric changes
   - Feature Attribution Monitor: Detects SHAP value changes

4. MONITORING SCHEDULE:
   - Minimum: Hourly
   - Typical: Daily
   - Uses captured inference data

5. REMEDIATION OPTIONS:
   - Retrain model
   - Roll back to previous version
   - Adjust feature engineering
   - Update business rules
""")

print(f"\n💡 Next step: python cleanup_monitoring.py")