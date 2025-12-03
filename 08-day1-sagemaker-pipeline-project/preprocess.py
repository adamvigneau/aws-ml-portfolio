"""
Data preprocessing script for SageMaker Processing
"""
import pandas as pd
import numpy as np
import argparse
import os

def preprocess_data(input_path, train_output_path, val_output_path):
    """
    Preprocess Titanic dataset
    """
    print("Loading data...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic features
    data = pd.DataFrame({
        'Pclass': np.random.choice([1, 2, 3], n_samples),
        'Sex': np.random.choice([0, 1], n_samples),
        'Age': np.random.randint(1, 80, n_samples),
        'SibSp': np.random.randint(0, 5, n_samples),
        'Parch': np.random.randint(0, 3, n_samples),
        'Fare': np.random.uniform(0, 100, n_samples),
        'Embarked': np.random.choice([0, 1, 2], n_samples)
    })
    
    # Generate target (simple rule-based for demo)
    # Make sure it's ONLY 0 or 1
    data['Survived'] = ((data['Pclass'] == 1) | 
                        (data['Sex'] == 1) | 
                        (data['Age'] < 18)).astype(int)
    
    # Add some randomness
    flip_indices = np.random.choice(n_samples, size=int(0.3 * n_samples), replace=False)
    data.loc[flip_indices, 'Survived'] = 1 - data.loc[flip_indices, 'Survived']
    
    # CRITICAL: Reorder so Survived is FIRST
    data = data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    
    print(f"Data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"Survived values: {sorted(data['Survived'].unique())}")
    print(f"Survival rate: {data['Survived'].mean():.2%}")
    
    # Verify label is 0/1
    assert data['Survived'].isin([0, 1]).all(), "Label must be 0 or 1!"
    
    # Split train/validation (80/20)
    train_data = data.sample(frac=0.8, random_state=42)
    val_data = data.drop(train_data.index)
    
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Save in XGBoost format (no header, label first)
    os.makedirs(train_output_path, exist_ok=True)
    os.makedirs(val_output_path, exist_ok=True)
    
    train_data.to_csv(
        os.path.join(train_output_path, 'train.csv'),
        index=False,
        header=False  # NO HEADER
    )
    
    val_data.to_csv(
        os.path.join(val_output_path, 'validation.csv'),
        index=False,
        header=False  # NO HEADER
    )
    
    print("✅ Preprocessing complete!")
    print(f"Saved to {train_output_path}/train.csv and {val_output_path}/validation.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data', type=str, default='/opt/ml/processing/input')
    parser.add_argument('--train-data', type=str, default='/opt/ml/processing/train')
    parser.add_argument('--val-data', type=str, default='/opt/ml/processing/validation')
    
    args = parser.parse_args()
    
    preprocess_data(args.input_data, args.train_data, args.val_data)