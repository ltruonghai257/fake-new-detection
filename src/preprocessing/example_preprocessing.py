"""
Complete Example of COOLANT Preprocessing Pipeline
This script demonstrates how to use the preprocessing pipelines for multimodal fake news detection
"""

import os
import json
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple

from text_preprocessing import TextPreprocessor
from image_preprocessing import ImagePreprocessor
from combined_preprocessing import CombinedPreprocessor, create_dataloaders
from data_utils import DataManager, DatasetSplitter, DataValidator, create_sample_data


def create_demo_dataset(num_samples: int = 100) -> Tuple[List[str], List[str], List[int]]:
    """
    Create a demonstration dataset with sample texts and images
    
    Args:
        num_samples: Number of samples to create
        
    Returns:
        Tuple of (texts, image_paths, labels)
    """
    print(f"Creating demo dataset with {num_samples} samples...")
    
    # Sample texts for fake news detection
    real_news_templates = [
        "Scientists at {} have discovered a new method for renewable energy generation.",
        "The government announced new policies to address climate change concerns.",
        "Local community in {} organizes charity event for homeless shelter.",
        "Researchers publish findings on medical breakthrough in cancer treatment.",
        "Economic report shows steady growth in the technology sector."
    ]
    
    fake_news_templates = [
        "BREAKING: {} cures all diseases overnight, doctors shocked!",
        "Celebrity spotted at {} - exclusive photos you won't believe!",
        "Scientists claim eating {} every day makes you live forever.",
        "Government hiding truth about {} - leaked documents reveal conspiracy.",
        "Miracle discovery: {} can solve all your problems in just 24 hours!"
    ]
    
    locations = ["Harvard", "MIT", "Stanford", "NASA", "CDC", "local hospital", "city hall"]
    objects = ["chocolate", "coffee", "vitamin C", "honey", "garlic", "green tea"]
    
    texts = []
    labels = []
    
    for i in range(num_samples):
        if i % 2 == 0:  # Real news
            template = np.random.choice(real_news_templates)
            location = np.random.choice(locations)
            text = template.format(location)
            label = 0  # Real news
        else:  # Fake news
            template = np.random.choice(fake_news_templates)
            if "celebrity" in template:
                location = np.random.choice(locations)
                text = template.format(location)
            else:
                obj = np.random.choice(objects)
                text = template.format(obj)
            label = 1  # Fake news
        
        texts.append(text)
        labels.append(label)
    
    # Create sample images
    image_dir = "./demo_images"
    os.makedirs(image_dir, exist_ok=True)
    
    image_paths = []
    for i in range(num_samples):
        image_path = os.path.join(image_dir, f"demo_image_{i}.jpg")
        
        # Create a random image if it doesn't exist
        if not os.path.exists(image_path):
            # Generate random image with different colors for real vs fake
            if labels[i] == 0:  # Real news - blueish images
                color = np.random.randint(100, 150, 3)
            else:  # Fake news - reddish images
                color = np.random.randint(150, 200, 3)
            
            # Create random image with the specified color tone
            image_array = np.random.randint(
                max(0, color[0] - 50), min(255, color[0] + 50), 
                (224, 224, 3), dtype=np.uint8
            )
            image_array[:, :, 1] = np.random.randint(
                max(0, color[1] - 50), min(255, color[1] + 50), 
                (224, 224), dtype=np.uint8
            )
            image_array[:, :, 2] = np.random.randint(
                max(0, color[2] - 50), min(255, color[2] + 50), 
                (224, 224), dtype=np.uint8
            )
            
            image = Image.fromarray(image_array)
            image.save(image_path)
        
        image_paths.append(image_path)
    
    print(f"Created {len(texts)} texts and {len(image_paths)} images")
    return texts, image_paths, labels


def example_text_preprocessing():
    """
    Example of text preprocessing pipeline
    """
    print("\n" + "="*50)
    print("TEXT PREPROCESSING EXAMPLE")
    print("="*50)
    
    # Sample texts
    texts = [
        "This is a real news article about scientific discoveries.",
        "FAKE NEWS: Eating chocolate cures cancer instantly!",
        "Government announces new environmental protection policies.",
        "BREAKING: Aliens landed in Central Park - witnesses say!"
    ]
    labels = [0, 1, 0, 1]
    
    # Initialize text preprocessor
    text_preprocessor = TextPreprocessor(
        model_name="bert-base-uncased",
        language="en",
        max_length=128
    )
    
    print("Original texts:")
    for i, text in enumerate(texts):
        print(f"{i+1}. {text}")
    
    print("\nCleaned texts:")
    for i, text in enumerate(texts):
        cleaned = text_preprocessor.clean_text(text)
        print(f"{i+1}. {cleaned}")
    
    # Extract features
    print("\nExtracting BERT features...")
    features, _ = text_preprocessor.preprocess_dataset(
        texts, labels,
        save_path="./example_outputs/text_features.pkl",
        extract_type="token_embeddings"
    )
    
    print(f"Text features shape: {features.shape}")
    print(f"Features dtype: {features.dtype}")
    
    # Validate features
    validator = DataValidator()
    report = validator.validate_text_features(features)
    print(f"Validation passed: {report['valid']}")
    if report['warnings']:
        print(f"Warnings: {report['warnings']}")


def example_image_preprocessing():
    """
    Example of image preprocessing pipeline
    """
    print("\n" + "="*50)
    print("IMAGE PREPROCESSING EXAMPLE")
    print("="*50)
    
    # Create sample images
    image_dir = "./example_images"
    os.makedirs(image_dir, exist_ok=True)
    
    image_paths = []
    for i in range(4):
        image_path = os.path.join(image_dir, f"example_image_{i}.jpg")
        
        if not os.path.exists(image_path):
            # Create random image
            image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            image = Image.fromarray(image_array)
            image.save(image_path)
        
        image_paths.append(image_path)
    
    labels = [0, 1, 0, 1]
    
    print(f"Processing {len(image_paths)} images...")
    
    # Initialize image preprocessor
    image_preprocessor = ImagePreprocessor(
        model_name="resnet18",
        image_size=(224, 224)
    )
    
    # Extract features
    print("Extracting ResNet features...")
    features, _ = image_preprocessor.preprocess_dataset(
        image_paths, labels,
        save_path="./example_outputs/image_features.pkl"
    )
    
    print(f"Image features shape: {features.shape}")
    print(f"Features dtype: {features.dtype}")
    
    # Validate features
    validator = DataValidator()
    report = validator.validate_image_features(features)
    print(f"Validation passed: {report['valid']}")
    if report['warnings']:
        print(f"Warnings: {report['warnings']}")


def example_combined_preprocessing():
    """
    Example of combined multimodal preprocessing
    """
    print("\n" + "="*50)
    print("COMBINED MULTIMODAL PREPROCESSING EXAMPLE")
    print("="*50)
    
    # Create demo dataset
    texts, image_paths, labels = create_demo_dataset(50)
    
    print(f"Dataset created with {len(texts)} samples")
    print(f"Real news: {sum(1 for l in labels if l == 0)}")
    print(f"Fake news: {sum(1 for l in labels if l == 1)}")
    
    # Initialize combined preprocessor
    combined_preprocessor = CombinedPreprocessor(
        text_model_name="bert-base-uncased",
        image_model_name="resnet18",
        language="en",
        max_text_length=128,
        image_size=(224, 224)
    )
    
    # Create train/val/test splits and preprocess
    print("\nCreating data splits and preprocessing...")
    datasets = combined_preprocessor.create_data_splits(
        texts, image_paths, labels,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        save_dir="./example_outputs/splits",
        save_format="pkl",
        random_seed=42
    )
    
    # Print dataset information
    for split_name, dataset in datasets.items():
        print(f"{split_name.upper()} set:")
        print(f"  Samples: {len(dataset)}")
        text_feat, img_feat, label = dataset[0]
        print(f"  Text feature shape: {text_feat.shape}")
        print(f"  Image feature shape: {img_feat.shape}")
        print(f"  Label: {label}")
    
    # Create DataLoaders
    print("\nCreating DataLoaders...")
    dataloaders = create_dataloaders(
        datasets, 
        batch_size=8, 
        num_workers=0  # Set to 0 for compatibility
    )
    
    for split_name, dataloader in dataloaders.items():
        print(f"{split_name.upper()} DataLoader:")
        batch_text, batch_image, batch_labels = next(iter(dataloader))
        print(f"  Batch text shape: {batch_text.shape}")
        print(f"  Batch image shape: {batch_image.shape}")
        print(f"  Batch labels shape: {batch_labels.shape}")
    
    return datasets, dataloaders


def example_data_management():
    """
    Example of data management utilities
    """
    print("\n" + "="*50)
    print("DATA MANAGEMENT EXAMPLE")
    print("="*50)
    
    # Initialize data manager
    manager = DataManager("./example_outputs/managed_data")
    
    # Save various data formats
    test_data = {
        'text_features': np.random.randn(10, 128, 768),
        'image_features': np.random.randn(10, 512),
        'labels': np.random.randint(0, 2, 10)
    }
    
    # Save as pickle
    pickle_path = manager.save_pickle(test_data, "multimodal_data.pkl")
    
    # Save as numpy
    npz_path = manager.save_npz(test_data, "multimodal_data.npz")
    
    # Save metadata
    metadata = {
        'num_samples': 10,
        'text_feature_dim': (128, 768),
        'image_feature_dim': 512,
        'num_classes': 2,
        'created_by': 'COOLANT preprocessing pipeline'
    }
    json_path = manager.save_json(metadata, "metadata.json")
    
    # List files
    files = manager.list_files()
    print(f"Files in managed data directory: {len(files)}")
    for file in files:
        info = manager.get_file_info(os.path.basename(file))
        print(f"  {info['name']}: {info['size_mb']:.2f} MB")
    
    # Load and verify data
    loaded_data = manager.load_pickle("multimodal_data.pkl")
    print(f"Loaded data keys: {list(loaded_data.keys())}")
    
    # Convert between formats
    from data_utils import convert_format
    convert_format(pickle_path, "./example_outputs/converted_data.npz", "pkl", "npz")


def example_validation():
    """
    Example of data validation
    """
    print("\n" + "="*50)
    print("DATA VALIDATION EXAMPLE")
    print("="*50)
    
    # Create sample data with some issues
    good_features = np.random.randn(100, 512).astype(np.float32)
    bad_features = np.array([[1, 2, np.nan, 4], [5, 6, 7, np.inf]]).astype(np.float32)
    good_labels = np.random.randint(0, 2, 100)
    bad_labels = np.array([0, 1, -1, 2])  # Contains invalid labels
    
    validator = DataValidator()
    
    # Validate good features
    print("Validating good features:")
    report = validator.validate_text_features(good_features)
    print(f"  Valid: {report['valid']}")
    print(f"  Shape: {report['shape']}")
    print(f"  Has NaN: {report['has_nan']}")
    print(f"  Has Inf: {report['has_inf']}")
    
    # Validate bad features
    print("\nValidating bad features:")
    report = validator.validate_image_features(bad_features)
    print(f"  Valid: {report['valid']}")
    print(f"  Warnings: {report['warnings']}")
    print(f"  Errors: {report['errors']}")
    
    # Validate good labels
    print("\nValidating good labels:")
    report = validator.validate_labels(good_labels)
    print(f"  Valid: {report['valid']}")
    print(f"  Classes: {report['num_classes']}")
    print(f"  Distribution: {report['class_distribution']}")
    
    # Validate bad labels
    print("\nValidating bad labels:")
    report = validator.validate_labels(bad_labels)
    print(f"  Valid: {report['valid']}")
    print(f"  Errors: {report['errors']}")


def main():
    """
    Main function to run all examples
    """
    print("COOLANT PREPROCESSING PIPELINE - COMPLETE EXAMPLE")
    print("=" * 60)
    
    # Create output directory
    os.makedirs("./example_outputs", exist_ok=True)
    
    try:
        # Run individual examples
        example_text_preprocessing()
        example_image_preprocessing()
        example_combined_preprocessing()
        example_data_management()
        example_validation()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("\nGenerated files:")
        output_dir = "./example_outputs"
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  {file_path}: {size_mb:.2f} MB")
        
        print("\nYou can now use the preprocessed data for COOLANT model training!")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
