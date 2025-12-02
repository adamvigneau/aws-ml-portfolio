"""
Model evaluation with debugging
"""
import json
import pathlib
import pickle
import tarfile
import sys
import os

def evaluate_model(model_path, test_data_path, output_path):
    """
    Evaluate XGBoost model with debugging
    """
    print("="*70)
    print("STARTING MODEL EVALUATION")
    print("="*70)
    
    # Import required packages
    import numpy as np
    import xgboost as xgb
    
    try:
        import pandas as pd
        has_pandas = True
    except:
        has_pandas = False
    
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        has_sklearn = True
    except:
        has_sklearn = False
    
    print(f"\nModel path: {model_path}")
    print(f"Test data path: {test_data_path}")
    
    # Extract and inspect model
    print("\n1. Extracting model...")
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)
    
    # Check file size
    file_size = os.path.getsize(model_path)
    print(f"   Model file size: {file_size} bytes")
    
    if file_size == 0:
        print("❌ Model file is empty!")
        sys.exit(1)
    
    # Extract
    with tarfile.open(model_path) as tar:
        print(f"   Files in archive:")
        for member in tar.getmembers():
            print(f"     - {member.name} ({member.size} bytes)")
        tar.extractall(path=".")
    
    # List extracted files
    print(f"\n   Extracted files in current directory:")
    for f in os.listdir("."):
        if os.path.isfile(f):
            size = os.path.getsize(f)
            print(f"     - {f} ({size} bytes)")
    
    # Try to find the model file
    model_file = None
    for filename in ['xgboost-model', 'model', 'xgboost_model']:
        if os.path.exists(filename):
            model_file = filename
            print(f"\n   Found model file: {model_file}")
            break
    
    if not model_file:
        print("\n❌ No model file found after extraction!")
        sys.exit(1)
    
    # Check model file size
    model_size = os.path.getsize(model_file)
    print(f"   Model file size: {model_size} bytes")
    
    if model_size == 0:
        print("❌ Model file is empty!")
        sys.exit(1)
    
    # Load model
    print("\n2. Loading model...")
    try:
        with open(model_file, "rb") as f:
            # Read first few bytes to check
            first_bytes = f.read(20)
            print(f"   First 20 bytes: {first_bytes}")
            f.seek(0)  # Reset to beginning
            
            # Try loading as XGBoost Booster directly
            try:
                model = xgb.Booster()
                model.load_model(model_file)
                print(f"   ✓ Loaded as XGBoost Booster")
            except:
                # Try pickle
                f.seek(0)
                model = pickle.load(f)
                print(f"   ✓ Loaded with pickle: {type(model)}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Load test data
    print("\n3. Loading test data...")
    if has_pandas:
        test_data = pd.read_csv(test_data_path, header=None)
        y_test = test_data.iloc[:, 0].values
        X_test = test_data.iloc[:, 1:].values
    else:
        test_data = np.genfromtxt(test_data_path, delimiter=',')
        y_test = test_data[:, 0].astype(int)
        X_test = test_data[:, 1:]
    
    print(f"   Test shape: {X_test.shape}")
    print(f"   Labels shape: {y_test.shape}")
    
    # Make predictions
    print("\n4. Making predictions...")
    dtest = xgb.DMatrix(X_test)
    
    # Check if model is Booster or needs different method
    if isinstance(model, xgb.Booster):
        predictions_proba = model.predict(dtest)
    else:
        predictions_proba = model.predict(dtest)
    
    predictions = (predictions_proba > 0.5).astype(int)
    print(f"   ✓ Predictions made: {predictions.shape}")
    
    # Calculate metrics
    print("\n5. Calculating metrics...")
    if has_sklearn:
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
    else:
        tp = np.sum((predictions == 1) & (y_test == 1))
        tn = np.sum((predictions == 0) & (y_test == 0))
        fp = np.sum((predictions == 1) & (y_test == 0))
        fn = np.sum((predictions == 0) & (y_test == 1))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "samples_evaluated": int(len(y_test))
    }
    
    print("\n6. Metrics:")
    for k, v in metrics.items():
        print(f"   {k}: {v}")
    
    # Save metrics
    print(f"\n7. Saving to {output_path}/evaluation.json")
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    with open(f"{output_path}/evaluation.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    
    return metrics

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='/opt/ml/processing/model/model.tar.gz')
    parser.add_argument('--test-data-path', type=str, default='/opt/ml/processing/test/validation.csv')
    parser.add_argument('--output-path', type=str, default='/opt/ml/processing/evaluation')
    
    args = parser.parse_args()
    
    try:
        evaluate_model(args.model_path, args.test_data_path, args.output_path)
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ ERROR: {e}")
        print(f"{'='*70}")
        import traceback
        traceback.print_exc()
        sys.exit(1)