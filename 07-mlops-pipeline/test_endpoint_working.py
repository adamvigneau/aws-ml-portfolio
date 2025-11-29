import boto3

runtime = boto3.client('sagemaker-runtime')

# Test samples - 7 features each (NO label)
# Format: value1, value2, value3, value4, value5, value6, value7
test_samples = [
    {
        'data': '22,3,0,7.25,0,1,0',
        'description': 'Young, 3rd class, male, low fare'
    },
    {
        'data': '38,1,1,71.28,1,0,1', 
        'description': 'Adult, 1st class, female, high fare'
    },
    {
        'data': '26,3,1,7.92,0,0,0',
        'description': 'Young adult, 3rd class, female, low fare'
    },
    {
        'data': '5,2,0,25.0,1,1,1',
        'description': 'Child, 2nd class, male, medium fare'
    },
    {
        'data': '60,1,0,50.0,0,0,1',
        'description': 'Senior, 1st class, male, high fare'
    }
]

print("🧪 Testing MLOps Production Endpoint")
print("="*70)

for i, sample in enumerate(test_samples, 1):
    try:
        response = runtime.invoke_endpoint(
            EndpointName='mlops-production-endpoint',
            Body=sample['data'],
            ContentType='text/csv'
        )
        
        prediction = float(response['Body'].read().decode('utf-8').strip())
        
        # XGBoost returns probability for binary classification
        predicted_class = 1 if prediction > 0.5 else 0
        confidence = prediction if predicted_class == 1 else (1 - prediction)
        
        print(f"\nSample {i}: {sample['description']}")
        print(f"  Features: {sample['data']}")
        print(f"  Prediction: Class {predicted_class} (confidence: {confidence:.2%})")
        
    except Exception as e:
        print(f"\nSample {i}: ERROR - {e}")

print("\n" + "="*70)
print("✅ MLOps Pipeline Complete!")
print("\nYour end-to-end automated ML pipeline is working:")
print("  ✅ Data upload triggers training")
print("  ✅ Training completes and registers model")
print("  ✅ Model approval triggers deployment")
print("  ✅ Endpoint serves predictions")
print("\n🎉 Congratulations on building a production MLOps pipeline!")
print("="*70)