# ViFactCheck Data Processing Modules

This document describes the comprehensive preprocessing and dataloader modules created for handling ViFactCheck dataset format.

## Overview

The ViFactCheck modules provide a complete pipeline for processing multimodal fake news detection data with the following structure:
- JSON entries contain an `images` array
- Each image object has `caption` (text) and `folder_path` (image path)
- Images are organized in subdirectories by news source (e.g., `thanh_nien/`, `dan_tri/`)

## Modules

### 1. `vifactcheck_preprocess.py`

Comprehensive preprocessing module for ViFactCheck data.

#### Key Features:
- **Text Cleaning**: Removes HTML tags, normalizes whitespace, handles Vietnamese diacritics
- **Image Validation**: Checks file existence, format, and readability
- **Source Extraction**: Automatically extracts news source from folder paths
- **Data Splitting**: Creates train/validation/test splits with stratification
- **Statistics**: Comprehensive dataset statistics and reporting

#### Main Classes:

##### `ViFactCheckPreprocessor`
```python
preprocessor = ViFactCheckPreprocessor(
    json_path="path/to/news_data_vifactcheck_train.json",
    image_base_dir="path/to/src/data/jpg",
    output_dir="preprocessed_data",
    min_text_length=10,
    max_text_length=2000
)

# Preprocess entire dataset
samples = preprocessor.preprocess_dataset()

# Create train/val/test splits
train, val, test = preprocessor.create_train_val_test_split(samples)

# Save splits
preprocessor.save_splits(train, val, test)
```

#### Convenience Function:
```python
from processing.vifactcheck_preprocess import preprocess_vifactcheck_data

results = preprocess_vifactcheck_data(
    json_path="path/to/data.json",
    image_base_dir="path/to/images",
    output_dir="preprocessed",
    test_size=0.2,
    val_size=0.1
)
```

#### Output Files:
- `train_samples.json/csv`: Training data
- `val_samples.json/csv`: Validation data  
- `test_samples.json/csv`: Test data
- `preprocessing_stats.json`: Dataset statistics

### 2. `vifactcheck_dataloader.py`

PyTorch DataLoader implementation for ViFactCheck data.

#### Key Features:
- **Multimodal Processing**: Handles both text and image data
- **Flexible Caching**: Configurable image and text caching
- **Memory Efficiency**: Optional image preloading and smart memory management
- **Custom Collation**: Specialized batch processing for multimodal data
- **Label Mapping**: Flexible source-to-label mapping

#### Main Classes:

##### `ViFactCheckDataset`
```python
from processing.vifactcheck_dataloader import ViFactCheckDataset

dataset = ViFactCheckDataset(
    data_path="preprocessed/train_samples.json",
    image_base_dir="src/data/jpg",
    label_mapping={'fake_source': 1, 'real_source': 0},
    max_length=128,
    embed_dim=768,
    feature_dim=512,
    cache_images=True,
    preload_images=False
)
```

#### DataLoader Creation:
```python
from processing.vifactcheck_dataloader import create_vifactcheck_dataloaders

train_loader, val_loader, test_loader = create_vifactcheck_dataloaders(
    train_data_path="preprocessed/train_samples.json",
    val_data_path="preprocessed/val_samples.json", 
    test_data_path="preprocessed/test_samples.json",
    image_base_dir="src/data/jpg",
    batch_size=32,
    num_workers=4,
    label_mapping={'dan_tri': 1, 'thanh_nien': 0}  # Example mapping
)
```

#### Single DataLoader:
```python
from processing.vifactcheck_dataloader import create_single_vifactcheck_dataloader

dataloader = create_single_vifactcheck_dataloader(
    data_path="data.json",
    image_base_dir="images/",
    batch_size=16,
    shuffle=True
)
```

### 3. `vifactcheck_pipeline_example.py`

Complete example demonstrating the entire pipeline.

## Data Format

### Input Format (ViFactCheck JSON):
```json
[
  {
    "images": [
      {
        "caption": "Vietnamese text content for fake news detection",
        "folder_path": "jpg/thanh_nien/thanh_nien_abc123.jpg",
        "src_url": "https://example.com/image.jpg"
      }
    ],
    "url": "https://example.com/article",
    "other_urls": []
  }
]
```

### Processed Format:
```json
[
  {
    "text": "Cleaned Vietnamese text content",
    "image_path": "jpg/thanh_nien/thanh_nien_abc123.jpg",
    "source": "thanh_nien",
    "label": 0,
    "sample_id": "0_0",
    "text_length": 15,
    "original_entry_idx": 0,
    "image_idx": 0
  }
]
```

### DataLoader Output Batch:
```python
{
    'text_features': torch.Tensor,      # Shape: (batch_size, max_length, embed_dim)
    'image_features': torch.Tensor,     # Shape: (batch_size, feature_dim)
    'labels': torch.Tensor,             # Shape: (batch_size,)
    'texts': List[str],                 # Original text content
    'image_paths': List[str],           # Image file paths
    'sample_ids': List[str],            # Sample identifiers
    'sources': List[str],               # News sources
    'batch_size': int                   # Batch size
}
```

## Usage Examples

### Complete Pipeline:
```python
import sys
sys.path.append('src')

from processing.vifactcheck_preprocess import preprocess_vifactcheck_data
from processing.vifactcheck_dataloader import create_vifactcheck_dataloaders

# 1. Preprocess data
results = preprocess_vifactcheck_data(
    json_path="src/data/json/news_data_vifactcheck_train.json",
    image_base_dir="src/data/jpg",
    output_dir="preprocessed_vifactcheck"
)

# 2. Create dataloaders
label_mapping = {
    'thanh_nien': 0,    # Real news
    'vn_express': 0,    # Real news  
    'dan_tri': 1,       # Fake news (example)
    'bao_tin_tuc': 1    # Fake news (example)
}

train_loader, val_loader, test_loader = create_vifactcheck_dataloaders(
    train_data_path=results['file_paths']['train_json'],
    val_data_path=results['file_paths']['val_json'],
    test_data_path=results['file_paths']['test_json'],
    image_base_dir="src/data/jpg",
    batch_size=32,
    label_mapping=label_mapping
)

# 3. Use in training loop
for batch in train_loader:
    text_features = batch['text_features']      # (32, 128, 768)
    image_features = batch['image_features']    # (32, 512)
    labels = batch['labels']                    # (32,)
    
    # Your model training code here
    # outputs = model(text_features, image_features)
    # loss = criterion(outputs, labels)
```

### Custom Text/Image Processing:
```python
def custom_text_processor(text):
    # Your custom text processing
    # Return torch.Tensor of shape (max_length, embed_dim)
    pass

def custom_image_processor(image):
    # Your custom image processing  
    # Return torch.Tensor of shape (feature_dim,)
    pass

dataset = ViFactCheckDataset(
    data_path="data.json",
    image_base_dir="images/",
    text_processor=custom_text_processor,
    image_processor=custom_image_processor
)
```

## Configuration Options

### Preprocessing Configuration:
- `min_text_length`: Minimum text length to keep (default: 10)
- `max_text_length`: Maximum text length to keep (default: 2000)
- `image_size`: Target image size (default: (224, 224))
- `valid_image_extensions`: Valid image formats (default: ['.jpg', '.jpeg', '.png'])

### DataLoader Configuration:
- `max_length`: Maximum text sequence length (default: 128)
- `embed_dim`: Text embedding dimension (default: 768)
- `feature_dim`: Image feature dimension (default: 512)
- `cache_images`: Enable image caching (default: True)
- `cache_text`: Enable text caching (default: False)
- `preload_images`: Preload all images into memory (default: False)

### Label Mapping:
Define which news sources are fake/real:
```python
label_mapping = {
    'reliable_source1': 0,  # Real news
    'reliable_source2': 0,  # Real news
    'fake_source1': 1,      # Fake news
    'fake_source2': 1       # Fake news
}
```

## Performance Considerations

### Memory Usage:
- **Image Caching**: Speeds up training but uses more memory
- **Text Caching**: Less memory impact, good for repeated access
- **Image Preloading**: Fastest but highest memory usage
- **Batch Size**: Balance between memory and training efficiency

### Recommended Settings:

#### For Large Datasets (>10K samples):
```python
dataset = ViFactCheckDataset(
    cache_images=True,
    cache_text=False,
    preload_images=False,
    batch_size=32,
    num_workers=4
)
```

#### For Small Datasets (<1K samples):
```python
dataset = ViFactCheckDataset(
    cache_images=True,
    cache_text=True,
    preload_images=True,
    batch_size=16,
    num_workers=2
)
```

## Error Handling

The modules include comprehensive error handling:
- **Missing Images**: Replaced with dummy images, logged as warnings
- **Invalid Text**: Filtered out during preprocessing
- **Corrupted Files**: Handled gracefully with fallbacks
- **Memory Issues**: Configurable caching to manage memory usage

## Statistics and Monitoring

Both modules provide detailed statistics:
- **Dataset Size**: Total and valid samples
- **Text Statistics**: Length distribution, vocabulary stats
- **Source Distribution**: Samples per news source
- **Label Distribution**: Real vs fake news balance
- **Processing Errors**: Missing images, invalid text, etc.

## Integration with Existing Code

These modules are designed to work alongside your existing ViFactCheck processor:
- Compatible with existing `ViFactCheckTextProcessor` and `ViFactCheckImageProcessor`
- Can be used as drop-in replacements for existing dataloaders
- Maintains backward compatibility with existing model interfaces

## Troubleshooting

### Common Issues:

1. **"Image not found" warnings**:
   - Check `image_base_dir` path
   - Verify image files exist in expected locations
   - Check file permissions

2. **Memory errors**:
   - Reduce `batch_size`
   - Disable `preload_images`
   - Reduce `num_workers`

3. **Slow loading**:
   - Enable `cache_images`
   - Increase `num_workers`
   - Use SSD storage for images

4. **Label imbalance**:
   - Check `label_mapping` configuration
   - Review source distribution in statistics
   - Consider stratified sampling

## Future Enhancements

Potential improvements:
- **Distributed Loading**: Multi-GPU data loading support
- **Data Augmentation**: Built-in augmentation for images and text
- **Online Processing**: Stream processing for very large datasets
- **Advanced Caching**: Disk-based caching for massive datasets
- **Metadata Support**: Additional metadata fields and filtering
