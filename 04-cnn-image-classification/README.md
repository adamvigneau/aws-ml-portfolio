## Week 4 Project: Image Classification with Convolutional Neural Network

### Overview
Built and trained a Convolutional Neural Network (CNN) using TensorFlow on AWS SageMaker to classify handwritten digits from the MNIST dataset. Compared deep learning performance against traditional machine learning (XGBoost) to understand when each approach is appropriate.

### Project Goal
Demonstrate understanding of:
- Deep learning fundamentals (neural networks, CNNs)
- Training deep learning models on AWS SageMaker
- When to use deep learning vs traditional ML
- Adapting to resource constraints in cloud environments

---

## Architecture
```
MNIST Dataset (60K images)
    ↓
Data Preprocessing (normalize, reshape)
    ↓
Upload to S3
    ↓
SageMaker Training Job (TensorFlow 2.12)
    ├─ Instance: ml.m5.xlarge (CPU)
    ├─ Framework: TensorFlow + Keras
    └─ Training: ~25 minutes
    ↓
Trained CNN Model (S3)
    ↓
Evaluation & Comparison
```

---

## Dataset: MNIST

- **What:** Handwritten digit images (0-9)
- **Size:** 60,000 training images, 10,000 test images
- **Image dimensions:** 28×28 pixels, grayscale
- **Classes:** 10 (digits 0-9)
- **Use case:** Classic computer vision benchmark, similar to real-world OCR applications

---

## CNN Architecture
```python
Input (28×28×1 grayscale image)
    ↓
Conv2D(32 filters, 3×3) + ReLU
    ↓
MaxPooling2D(2×2)
    ↓
Conv2D(64 filters, 3×3) + ReLU
    ↓
MaxPooling2D(2×2)
    ↓
Conv2D(64 filters, 3×3) + ReLU
    ↓
Flatten
    ↓
Dense(64) + ReLU
    ↓
Dropout(0.5)
    ↓
Dense(10) + Softmax
    ↓
Output (10 class probabilities)
```

**Key Features:**
- **3 Convolutional Layers:** Learn spatial features (edges → shapes → digits)
- **2 MaxPooling Layers:** Reduce dimensions, increase robustness
- **Dropout Layer:** Prevent overfitting (50% dropout rate)
- **Total Parameters:** ~1.2 million trainable parameters

---

## Training Configuration

### Hyperparameters
- **Optimizer:** Adam (adaptive learning rate)
- **Learning Rate:** 0.001
- **Batch Size:** 64
- **Epochs:** 5
- **Loss Function:** Sparse Categorical Cross-Entropy
- **Validation Split:** 10%

### AWS Infrastructure
- **Service:** Amazon SageMaker Training Jobs
- **Framework:** TensorFlow 2.12
- **Python Version:** 3.10
- **Instance Type:** ml.m5.xlarge (CPU)
- **Training Time:** ~25 minutes
- **Cost:** ~$0.10 total

### Technical Constraints
**GPU Quota Limitation:**
- Account had service quota of 0 for GPU instances (ml.p3.2xlarge, ml.g4dn.xlarge)
- Common for new AWS accounts to prevent unexpected costs
- Adapted by using CPU instance (ml.m5.xlarge)
- Trade-off: Longer training time (~25 min vs ~5-10 min on GPU) but same accuracy
- **Learning:** Resource constraints are real in production - adapted solution while maintaining performance

---

## Results

### CNN Performance
- **Test Accuracy:** 98-99%
- **Training Loss:** Converged smoothly over 5 epochs
- **Validation Accuracy:** ~98% (minimal overfitting)
- **Inference:** Correctly classifies unseen handwritten digits

### Model Comparison: CNN vs XGBoost

| Metric | CNN (Deep Learning) | XGBoost (Traditional ML) |
|--------|---------------------|--------------------------|
| **Test Accuracy** | 98-99% | 96-97% |
| **Training Time** | ~25 min | ~4 min |
| **Instance Type** | ml.m5.xlarge | ml.m5.xlarge |
| **Model Size** | ~5 MB | ~2 MB |
| **Understands Spatial Structure** | ✅ Yes | ❌ No |
| **Parameter Count** | ~1.2M parameters | Tree-based (different paradigm) |
| **Best Use Case** | Images, spatial data | Tabular, structured data |

---

## Key Insights

### Why CNN Wins for Images

1. **Spatial Understanding**
   - CNNs preserve 2D structure of images
   - Convolutional filters detect local patterns (edges, curves, shapes)
   - XGBoost flattens image to 1D array, losing spatial relationships

2. **Feature Learning**
   - CNNs automatically learn hierarchical features
   - Layer 1: Edges and simple shapes
   - Layer 2: Complex patterns and textures
   - Layer 3: High-level features (digit shapes)
   - No manual feature engineering required

3. **Translation Invariance**
   - Max pooling makes model robust to small shifts/rotations
   - Same digit recognized regardless of position in image

4. **Parameter Efficiency**
   - Weight sharing in convolutional layers
   - Fewer parameters than fully-connected network
   - Reduces overfitting on image data

### When to Use What

**Use CNN (Deep Learning) when:**
- ✅ Working with images, video, or spatial data
- ✅ Have large datasets (10,000+ examples)
- ✅ Need to learn complex patterns automatically
- ✅ Accuracy is critical
- ✅ Have GPU access (ideal but not required)

**Use XGBoost (Traditional ML) when:**
- ✅ Working with tabular/structured data (CSVs, databases)
- ✅ Have smaller datasets (<10,000 examples)
- ✅ Need interpretability (feature importance)
- ✅ Limited compute resources
- ✅ Fast training is priority
- ✅ Data doesn't have spatial/sequential structure

---

## Technical Implementation Details

### Data Preprocessing Pipeline
```python
# Normalization
x_train = x_train.astype('float32') / 255.0  # Scale pixels to [0, 1]

# Reshape for CNN input
x_train = x_train.reshape(-1, 28, 28, 1)  # Add channel dimension

# Upload to S3 for SageMaker access
sess.upload_data('train_data.npy', bucket=bucket, key_prefix='mnist-cnn/data')
```

### SageMaker Training Script Structure

**mnist_cnn.py** - Custom training script:
- Accepts hyperparameters via command-line arguments
- Loads training data from SageMaker input channels
- Builds CNN model using Keras Sequential API
- Trains with validation split for monitoring
- Saves model to SageMaker model directory
- Integrates with SageMaker's managed training infrastructure

### Regularization Techniques Used

1. **Dropout (0.5):** Randomly drops 50% of neurons during training
2. **Validation Split:** 10% of training data held out for validation
3. **Early stopping consideration:** Could monitor validation loss
4. **Max Pooling:** Reduces spatial dimensions, adds robustness

---

## Project Structure
```
week-4-cnn-mnist/
├── notebook.ipynb           # Main notebook with all code
├── mnist_cnn.py            # SageMaker training script
├── README.md               # This file
└── architecture_diagram.png # Visual architecture (optional)
```

---

## What This Project Demonstrates

### Deep Learning Knowledge
- ✅ Understanding of CNN architecture and components
- ✅ Knowledge of activation functions (ReLU, Softmax)
- ✅ Regularization techniques (Dropout)
- ✅ Loss functions and optimizers (Cross-Entropy, Adam)
- ✅ Training monitoring and validation

### AWS/SageMaker Skills
- ✅ SageMaker training jobs with custom scripts
- ✅ TensorFlow framework integration
- ✅ S3 data management for ML
- ✅ IAM roles and permissions
- ✅ Resource constraint adaptation

### ML Engineering Mindset
- ✅ Comparing multiple approaches (CNN vs XGBoost)
- ✅ Understanding trade-offs (accuracy vs cost vs time)
- ✅ Adapting to constraints (GPU quota limitations)
- ✅ Making practical decisions based on problem type
- ✅ Production considerations (training time, cost, scalability)

### Data Engineer → ML Engineer Transition
- ✅ Applied DE skills: data preprocessing, S3 management
- ✅ Learned new domain: deep learning architectures
- ✅ Integrated both: built end-to-end ML pipeline
- ✅ Practical approach: chose appropriate tools for the task

---

## Lessons Learned

### Technical
1. **CNNs are specialized for spatial data** - dramatically outperform traditional ML on images
2. **CPU training is viable for smaller models** - MNIST trained reasonably fast on CPU
3. **SageMaker abstracts infrastructure** - focus on model, not servers
4. **Hyperparameter tuning matters** - batch size and learning rate significantly impact training

### Operational
1. **Account quotas are real** - plan for GPU access or alternatives
2. **Cost awareness** - CPU instances cost ~35x less than P3 GPUs
3. **Training time trade-offs** - CPU took 5x longer but cost 1/35th
4. **Flexibility is key** - adapted quickly when GPU quota blocked

### Conceptual
1. **Right tool for the job** - CNNs for images, XGBoost for tabular
2. **Deep learning isn't always better** - XGBoost still best for structured data
3. **Feature engineering matters less** - CNNs learn features automatically
4. **More parameters ≠ better** - CNN has more params but better suited for task

---

## Future Enhancements

If continuing this project, could explore:

1. **Hyperparameter Tuning**
   - Use SageMaker Automatic Model Tuning
   - Experiment with different architectures (more/fewer layers)
   - Try different optimizers (SGD with momentum, RMSprop)

2. **Model Deployment**
   - Deploy to SageMaker endpoint for real-time predictions
   - Build simple web interface for digit recognition
   - Test inference latency and throughput

3. **Advanced Techniques**
   - Data augmentation (rotation, shifting, scaling)
   - Transfer learning from pre-trained models
   - Ensemble methods (combine multiple CNNs)

4. **More Complex Datasets**
   - CIFAR-10 (color images, 10 classes)
   - Fashion-MNIST (clothing items)
   - Custom dataset (real-world business problem)

5. **Model Optimization**
   - Quantization for smaller model size
   - Pruning to reduce parameters
   - Deploy to edge devices with SageMaker Neo

---

## Comparison to Previous Weeks

### Week 1: Local ML Training
- Built models in local Jupyter notebook
- Simple preprocessing and training
- Focus: Understanding ML fundamentals

### Week 2: SageMaker with Traditional ML
- Deployed XGBoost to SageMaker
- Learned SageMaker workflow
- Focus: Cloud ML deployment

### Week 3: Production Data Pipelines
- Built AWS Glue ETL pipeline
- Feature engineering at scale
- Focus: Data engineering for ML

### Week 4: Deep Learning (This Project)
- Trained neural networks on SageMaker
- Understood when to use deep learning
- Focus: CNNs and framework integration
- **Brings together:** Data prep (Week 3) + SageMaker (Week 2) + Advanced algorithms

---

## Resources Used

### Documentation
- [AWS SageMaker TensorFlow](https://docs.aws.amazon.com/sagemaker/latest/dg/tf.html)
- [TensorFlow Keras API](https://www.tensorflow.org/guide/keras)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

### Learning Materials
- 3Blue1Brown Neural Networks series
- AWS Machine Learning documentation
- TensorFlow tutorials

---

## Conclusion

This project successfully demonstrated:
- Building and training CNNs from scratch
- Understanding deep learning architecture design
- Using AWS SageMaker for deep learning workloads
- Comparing deep learning vs traditional ML
- Adapting to real-world constraints (GPU quotas)
- Making informed decisions about algorithm selection

**Key Takeaway:** Deep learning (CNNs) excels at image/spatial data, while traditional ML (XGBoost) remains superior for tabular data. Choosing the right tool for the problem type is more important than always using the newest/fanciest technique.

---

**Training Cost:** ~$0.10
**Training Time:** ~25 minutes
**Final Accuracy:** 98-99%
**Status:** ✅ Complete