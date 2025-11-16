# COOLANT Preprocessing Module

This repository provides a comprehensive preprocessing module for multimodal fake news detection, based on the [COOLANT](https://github.com/wishever/COOLANT) repository. The module handles both text and image preprocessing with options to save processed data in pickle or numpy formats.

## Module Structure

```
src/
└── preprocessing/
    ├── __init__.py                    # Module initialization and convenience functions
    ├── text_preprocessing.py          # BERT-based text preprocessing
    ├── image_preprocessing.py         # ResNet-based image preprocessing
    ├── combined_preprocessing.py      # Integrated multimodal preprocessing
    ├── data_utils.py                  # Data management and validation utilities
    └── example_preprocessing.py       # Comprehensive examples
```

## Features

- **Text Preprocessing**: BERT-based tokenization and feature extraction for English and Chinese text
- **Image Preprocessing**: ResNet-based feature extraction with standard transforms
- **Combined Pipeline**: Integrated multimodal preprocessing with coordinated saving
- **Data Management**: Utilities for saving, loading, and validating preprocessed data
- **Flexible Formats**: Support for both pickle (.pkl) and numpy (.npz) formats
- **Dataset Splits**: Automatic train/validation/test splitting with stratification

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd fake-new-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Import the Preprocessing Module

```python
# Add src to path and import main classes
import sys
sys.path.append('./src')
from preprocessing import TextPreprocessor, ImagePreprocessor, CombinedPreprocessor

# Or use convenience functions
from preprocessing import preprocess_text_dataset, preprocess_image_dataset, preprocess_multimodal_dataset
```

### 2. Basic Text Preprocessing

```python
# Sample data
texts = ["This is a news article.", "Another news story."]
labels = [0, 1]  # 0: real, 1: fake

# Quick preprocessing using convenience function
features, labels = preprocess_text_dataset(
    texts, labels,
    save_path="./processed_data/text_features.pkl",
    language="en"
)

print(f"Features shape: {features.shape}")
```

### 3. Basic Image Preprocessing

```python
# Sample data
image_paths = ["image1.jpg", "image2.jpg"]
labels = [0, 1]

# Quick preprocessing using convenience function
features, labels = preprocess_image_dataset(
    image_paths, labels,
    save_path="./processed_data/image_features.pkl"
)

print(f"Features shape: {features.shape}")
```

### 4. Combined Multimodal Preprocessing

```python
# Sample data
texts = ["News article 1", "News article 2"]
image_paths = ["image1.jpg", "image2.jpg"]
labels = [0, 1]

# Quick multimodal preprocessing with train/val/test splits
datasets = preprocess_multimodal_dataset(
    texts, image_paths, labels,
    save_dir="./processed_data",
    language="en"
)

# Access datasets
train_dataset = datasets['train']
val_dataset = datasets['val']
test_dataset = datasets['test']
```

### 5. Advanced Usage with Custom Configuration

```python
# Initialize custom preprocessor
preprocessor = CombinedPreprocessor(
    text_model_name="bert-base-uncased",
    image_model_name="resnet50",
    language="en",
    max_text_length=256
)

# Process with custom settings
datasets = preprocessor.create_data_splits(
    texts, image_paths, labels,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    save_dir="./custom_processed_data",
    save_format="npz"
)
```

## Detailed Usage

### Text Preprocessing Pipeline

The `TextPreprocessor` class handles:

- **Text Cleaning**: Removes punctuation, normalizes whitespace
- **Tokenization**: Uses BERT tokenizers for English/Chinese
- **Feature Extraction**: Extracts BERT embeddings or token-level features
- **Language Support**: English (bert-base-uncased) and Chinese (bert-base-chinese)

#### Key Methods:

- `clean_text(text)`: Clean individual text strings
- `tokenize_text(text)`: Tokenize text for BERT input
- `extract_bert_features(texts)`: Extract BERT pooled outputs
- `extract_token_embeddings(texts)`: Extract token-level embeddings (FastCNN compatible)
- `preprocess_dataset(texts, labels, save_path)`: Process complete datasets

### Image Preprocessing Pipeline

The `ImagePreprocessor` class handles:

- **Image Loading**: PIL-based image loading with error handling
- **Transforms**: Standard ResNet preprocessing (resize, normalize, etc.)
- **Feature Extraction**: ResNet18/ResNet50 feature extraction
- **Augmentation**: Optional data augmentation support

#### Key Methods:

- `load_and_preprocess_image(path)`: Load and preprocess single image
- `extract_features(image_paths)`: Extract features from image paths
- `extract_features_from_images(images)`: Extract features from PIL Images
- `preprocess_dataset(image_paths, labels, save_path)`: Process complete datasets

### Combined Multimodal Pipeline

The `CombinedPreprocessor` class coordinates both text and image preprocessing:

- **Synchronized Processing**: Ensures text-image pairs are processed together
- **Dataset Creation**: Creates PyTorch-compatible datasets
- **Split Management**: Handles train/val/test splits with metadata
- **Format Flexibility**: Saves in pickle or numpy formats

#### Key Methods:

- `preprocess_sample(text, image_path)`: Process single text-image pair
- `preprocess_dataset(texts, image_paths, labels, save_dir)`: Process complete dataset
- `create_data_splits(...)`: Create and preprocess train/val/test splits

### Data Management Utilities

The `data_utils` module provides:

- **DataManager**: File I/O operations for different formats
- **DatasetSplitter**: Stratified dataset splitting
- **DataValidator**: Data quality validation
- **Format Conversion**: Between pickle and numpy formats

## File Formats

### Pickle Format (.pkl)
- Stores Python objects directly
- Preserves data types and structure
- Good for complex nested data

### Numpy Format (.npz)
- Stores numpy arrays efficiently
- Cross-platform compatibility
- Better for large numerical data

### Example File Structure
```
processed_data/
├── train/
│   ├── text_features.pkl
│   ├── image_features.pkl
│   ├── combined_dataset.pkl
│   └── train_metadata.json
├── val/
│   └── ...
├── test/
│   └── ...
└── metadata.json
```

## Dataset Compatibility

### Twitter Dataset (English)
```python
from combined_preprocessing import preprocess_twitter_dataset

datasets = preprocess_twitter_dataset(
    data_path="./twitter_data.json",
    image_dir="./twitter_images/",
    save_dir="./processed_data/twitter"
)
```

### Weibo Dataset (Chinese)
```python
from combined_preprocessing import preprocess_weibo_dataset

datasets = preprocess_weibo_dataset(
    data_path="./weibo_data.json", 
    image_dir="./weibo_images/",
    save_dir="./processed_data/weibo"
)
```

## Data Validation

The preprocessing pipelines include comprehensive validation:

```python
from data_utils import DataValidator

validator = DataValidator()

# Validate text features
text_report = validator.validate_text_features(features)
print(f"Text validation: {text_report['valid']}")

# Validate image features  
image_report = validator.validate_image_features(features)
print(f"Image validation: {image_report['valid']}")

# Validate labels
label_report = validator.validate_labels(labels)
print(f"Label validation: {label_report['valid']}")
```

## Running Examples

### Simple Example (Main Directory)

Execute the simple example script:

```bash
python preprocessing_example.py
```

This will demonstrate basic usage of the preprocessing module with convenience functions.

### Comprehensive Examples (Module Directory)

Execute the complete example script within the preprocessing module:

```bash
python src/preprocessing/example_preprocessing.py
```

This will:
1. Create sample demonstration data
2. Run text preprocessing example
3. Run image preprocessing example
4. Run combined multimodal preprocessing
5. Demonstrate data management utilities
6. Show data validation in action

## Integration with COOLANT Training

The preprocessed data is fully compatible with the original COOLANT training pipeline:

```python
# Load preprocessed data for COOLANT training
from combined_preprocessing import CombinedPreprocessor

text_features, image_features, labels = CombinedPreprocessor.load_combined_dataset(
    "./processed_data/train/combined_dataset.pkl"
)

# Create COOLANT-compatible dataset
from torch.utils.data import DataLoader
dataset = MultimodalDataset(text_features, image_features, labels)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Use in COOLANT training loop
for batch_text, batch_image, batch_labels in dataloader:
    # Pass to COOLANT model
    pass
```

## Configuration Options

### Text Preprocessing Options
- `model_name`: BERT model ("bert-base-uncased", "bert-base-chinese")
- `max_length`: Maximum sequence length (default: 512)
- `language`: Language code ("en" or "zh")
- `device`: Processing device ("cuda" or "cpu")

### Image Preprocessing Options
- `model_name`: Backbone model ("resnet18" or "resnet50")
- `pretrained`: Use pretrained weights (default: True)
- `image_size`: Target image size (default: (224, 224))
- `device`: Processing device ("cuda" or "cpu")

### Combined Pipeline Options
- `save_format`: Output format ("pkl" or "npz")
- `batch_size`: Processing batch size (default: 32)
- `train_ratio/val_ratio/test_ratio`: Dataset split ratios

## Performance Tips

1. **GPU Acceleration**: Use CUDA for faster preprocessing
2. **Batch Processing**: Process data in batches for efficiency
3. **Memory Management**: Use numpy format for large datasets
4. **Parallel Processing**: Set `num_workers > 0` in DataLoaders

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Image Loading Errors**: Check image paths and formats
3. **Tokenization Errors**: Verify text encoding and language settings
4. **File Permission Errors**: Ensure write permissions for save directories

### Error Handling

The pipelines include robust error handling:
- Missing images are replaced with zero tensors
- Invalid text is cleaned and processed when possible
- Comprehensive validation reports help identify issues

## Citation

If you use this preprocessing pipeline, please cite the original COOLANT paper:

```bibtex
@inproceedings{10.1145/3581783.3613850,
  author = {Wang, Longzheng and Zhang, Chuang and Xu, Hongbo and Xu, Yongxiu and Xu, Xiaohan and Wang, Siqi},
  title = {Cross-Modal Contrastive Learning for Multimodal Fake News Detection},
  year = {2023},
  isbn = {9798400701085},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3581783.3613850},
  doi = {10.1145/3581783.3613850},
  booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
  pages = {5696–5704},
  numpages = {9},
  keywords = {social media, multimodal fusion, fake news detection, contrastive learning},
  location = {Ottawa ON, Canada},
  series = {MM '23}
}
```

## License

This code is based on the COOLANT repository and follows the same licensing terms.
