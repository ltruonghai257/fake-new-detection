"""
Simple example demonstrating the COOLANT preprocessing module usage
"""

import os
import sys
import numpy as np
from PIL import Image

# Add src to path to import preprocessing module
sys.path.append('./src')

# Import from the preprocessing module in src/
from preprocessing import (
    TextPreprocessor, 
    ImagePreprocessor, 
    CombinedPreprocessor,
    DataManager,
    preprocess_text_dataset,
    preprocess_image_dataset,
    preprocess_multimodal_dataset
)

def create_sample_data():
    """Create sample data for demonstration"""
    print("Creating sample data...")
    
    # Sample texts
    texts = [
        "Scientists discover new renewable energy source.",
        "BREAKING: Chocolate cures all diseases instantly!",
        "Government announces new environmental policies.",
        "Celebrity spotted at local restaurant - exclusive photos!"
    ]
    
    # Create sample images
    image_dir = "./sample_images"
    os.makedirs(image_dir, exist_ok=True)
    
    image_paths = []
    for i in range(len(texts)):
        image_path = os.path.join(image_dir, f"sample_{i}.jpg")
        if not os.path.exists(image_path):
            # Create random image
            image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            image = Image.fromarray(image_array)
            image.save(image_path)
        image_paths.append(image_path)
    
    # Labels (0: real news, 1: fake news)
    labels = [0, 1, 0, 1]
    
    return texts, image_paths, labels

def example_basic_usage():
    """Example of basic preprocessing module usage"""
    print("\n" + "="*50)
    print("BASIC PREPROCESSING MODULE USAGE")
    print("="*50)
    
    texts, image_paths, labels = create_sample_data()
    
    # Quick text preprocessing
    print("Preprocessing text data...")
    text_features, processed_labels = preprocess_text_dataset(
        texts, labels,
        save_path="./processed_data/text_features.pkl",
        language="en"
    )
    print(f"Text features shape: {text_features.shape}")
    
    # Quick image preprocessing
    print("Preprocessing image data...")
    image_features, _ = preprocess_image_dataset(
        image_paths, labels,
        save_path="./processed_data/image_features.pkl"
    )
    print(f"Image features shape: {image_features.shape}")
    
    # Quick multimodal preprocessing
    print("Preprocessing multimodal data...")
    datasets = preprocess_multimodal_dataset(
        texts, image_paths, labels,
        save_dir="./processed_data/multimodal",
        language="en"
    )
    
    print(f"Created datasets: {list(datasets.keys())}")
    for split_name, dataset in datasets.items():
        print(f"  {split_name}: {len(dataset)} samples")

def example_advanced_usage():
    """Example of advanced preprocessing with custom configurations"""
    print("\n" + "="*50)
    print("ADVANCED PREPROCESSING USAGE")
    print("="*50)
    
    texts, image_paths, labels = create_sample_data()
    
    # Initialize custom text preprocessor
    text_preprocessor = TextPreprocessor(
        model_name="bert-base-uncased",
        max_length=256,
        language="en",
        device="cuda" if os.system("nvidia-smi") == 0 else "cpu"
    )
    
    # Initialize custom image preprocessor
    image_preprocessor = ImagePreprocessor(
        model_name="resnet50",
        image_size=(224, 224),
        device="cuda" if os.system("nvidia-smi") == 0 else "cpu"
    )
    
    # Initialize combined preprocessor with custom settings
    combined_preprocessor = CombinedPreprocessor(
        text_model_name="bert-base-uncased",
        image_model_name="resnet50",
        language="en",
        max_text_length=256
    )
    
    # Process with custom settings
    print("Processing with custom configurations...")
    datasets = combined_preprocessor.create_data_splits(
        texts, image_paths, labels,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        save_dir="./processed_data/advanced",
        save_format="npz"
    )
    
    print("Advanced preprocessing completed!")

def example_data_management():
    """Example of data management utilities"""
    print("\n" + "="*50)
    print("DATA MANAGEMENT EXAMPLE")
    print("="*50)
    
    # Initialize data manager
    manager = DataManager("./managed_data")
    
    # Save sample data
    sample_data = {
        'features': np.random.randn(10, 512),
        'labels': np.random.randint(0, 2, 10),
        'metadata': {
            'num_samples': 10,
            'feature_dim': 512,
            'num_classes': 2
        }
    }
    
    # Save in different formats
    manager.save_pickle(sample_data, "sample_data.pkl")
    manager.save_npz(sample_data, "sample_data.npz")
    
    # List saved files
    files = manager.list_files()
    print(f"Saved files: {len(files)}")
    
    # Load and verify
    loaded_data = manager.load_pickle("sample_data.pkl")
    print(f"Loaded data keys: {list(loaded_data.keys())}")

if __name__ == "__main__":
    print("COOLANT PREPROCESSING MODULE EXAMPLE")
    print("=" * 60)
    
    # Create output directories
    os.makedirs("./processed_data", exist_ok=True)
    os.makedirs("./managed_data", exist_ok=True)
    
    try:
        # Run examples
        example_basic_usage()
        example_advanced_usage()
        example_data_management()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("\nModule structure:")
        print("preprocessing/")
        print("├── __init__.py")
        print("├── text_preprocessing.py")
        print("├── image_preprocessing.py")
        print("├── combined_preprocessing.py")
        print("├── data_utils.py")
        print("└── example_preprocessing.py")
        
        print("\nUsage examples:")
        print("from preprocessing import TextPreprocessor")
        print("from preprocessing import preprocess_multimodal_dataset")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
