import tensorflow as tf
from tensorflow import keras
import argparse
import os
import numpy as np

def create_cnn_model():
    """Create a simple CNN for MNIST"""
    model = keras.Sequential([
        # Convolutional layers
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Fully connected layers
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')  # 10 classes
    ])
    
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    
    # SageMaker parameters
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    
    args, _ = parser.parse_known_args()
    
    # Load data
    print("Loading training data...")
    x_train = np.load(os.path.join(args.train, 'train_data.npy'))
    y_train = np.load(os.path.join(args.train, 'train_labels.npy'))
    
    print(f"Training data shape: {x_train.shape}")
    
    # Create model
    print("Creating CNN model...")
    model = create_cnn_model()
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    
    # Train model
    print(f"Training for {args.epochs} epochs...")
    history = model.fit(
        x_train, y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=0.1,
        verbose=1
    )
    
    # Save model
    print(f"Saving model to {args.model_dir}...")
    model.save(os.path.join(args.model_dir, 'mnist_cnn_model'))
    
    print("✅ Training complete!")