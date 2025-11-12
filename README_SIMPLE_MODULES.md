# Simple ViFactCheck Processing Modules

Clean, focused modules for processing ViFactCheck dataset with separate text and image processing.

## Modules Overview

### 1. `text_processor.py` - Text Processing
- **Purpose**: Vietnamese text cleaning, tokenization, and feature extraction
- **Key Features**:
  - Text cleaning (HTML removal, normalization)
  - PhoBERT tokenization support
  - Feature extraction with fallback options
  - Batch processing capability

### 2. `image_processor.py` - Image Processing  
- **Purpose**: Image loading, preprocessing, and feature extraction
- **Key Features**:
  - Image validation and loading
  - ResNet/EfficientNet feature extraction
  - Image augmentation support
  - Batch processing capability

### 3. `simple_preprocess.py` - Data Preprocessing
- **Purpose**: Clean ViFactCheck data preprocessing
- **Key Features**:
  - Raw ViFactCheck JSON parsing
  - Text and image validation
  - Train/val/test splitting
  - Statistics reporting

### 4. `simple_dataloader.py` - PyTorch DataLoader
- **Purpose**: Simple, efficient dataloader for training
- **Key Features**:
  - Separate text and image processing
  - Flexible label mapping
  - Clean batch collation
  - Memory efficient

## Quick Usage

### Complete Pipeline
```python
import sys
sys.path.append('src')

from processing.simple_preprocess import preprocess_vifactcheck
from processing.simple_dataloader import create_train_val_test_loaders

# 1. Preprocess data
results = preprocess_vifactcheck(
    json_path="src/data/json/news_data_vifactcheck_train.json",
    image_base_dir="src/data/jpg",
    output_dir="preprocessed"
)

# 2. Create dataloaders
label_mapping = {
    'thanh_nien': 0,    # Real news
    'vn_express': 0,    # Real news
    'dan_tri': 1,       # Fake news
    'bao_tin_tuc': 1    # Fake news
}

train_loader, val_loader, test_loader = create_train_val_test_loaders(
    train_path=results['files']['train'],
    val_path=results['files']['val'],
    test_path=results['files']['test'],
    image_base_dir="src/data/jpg",
    batch_size=32,
    label_mapping=label_mapping
)

# 3. Use in training
for batch in train_loader:
    text_features = batch['text_features']    # (32, 768)
    image_features = batch['image_features']  # (32, 512)
    labels = batch['labels']                  # (32,)
    
    # Your model training code here
```

### Individual Processors
```python
from processing.text_processor import TextProcessor
from processing.image_processor import ImageProcessor

# Text processing
text_processor = TextProcessor(max_length=128)
text_result = text_processor.process("Vietnamese text here")
print(f"Features shape: {text_result['features'].shape}")

# Image processing  
image_processor = ImageProcessor(feature_dim=512)
image_result = image_processor.process("path/to/image.jpg")
print(f"Features shape: {image_result['features'].shape}")
```

## Data Format

### Input (ViFactCheck JSON):
```json
[
  {
    "images": [
      {
        "caption": "Vietnamese text content",
        "folder_path": "jpg/source_name/image.jpg"
      }
    ]
  }
]
```

### Output Batch:
```python
{
    'text_features': torch.Tensor,     # (batch_size, 768)
    'text_tokens': torch.Tensor,       # (batch_size, max_length)
    'text_mask': torch.Tensor,         # (batch_size, max_length)
    'image_features': torch.Tensor,    # (batch_size, feature_dim)
    'image_tensors': torch.Tensor,     # (batch_size, 3, 224, 224)
    'labels': torch.Tensor,            # (batch_size,)
    'texts': List[str],                # Original texts
    'image_paths': List[str],          # Image paths
    'sources': List[str],              # News sources
    'sample_ids': List[str]            # Sample IDs
}
```

## Configuration

### Text Processor:
- `max_length`: Maximum sequence length (default: 128)
- `tokenizer_name`: Pretrained tokenizer (default: "vinai/phobert-base")

### Image Processor:
- `image_size`: Target size (default: (224, 224))
- `feature_dim`: Output dimension (default: 512)
- `model_name`: Pretrained model (default: "resnet50")

### Label Mapping:
```python
label_mapping = {
    'reliable_source': 0,  # Real news
    'fake_source': 1       # Fake news
}
```

## Example Script

Run the complete example:
```bash
python examples/simple_pipeline.py
```

## Key Benefits

✅ **Separate Processing**: Independent text and image processors  
✅ **Clean Code**: Focused, readable implementations  
✅ **Memory Efficient**: No unnecessary caching or complexity  
✅ **Flexible**: Easy to customize and extend  
✅ **Vietnamese Support**: Proper handling of Vietnamese text  
✅ **Error Handling**: Robust error handling with fallbacks  

## File Structure
```
src/processing/
├── text_processor.py      # Text processing
├── image_processor.py     # Image processing  
├── simple_preprocess.py   # Data preprocessing
└── simple_dataloader.py   # PyTorch dataloader

examples/
└── simple_pipeline.py     # Complete example
```

This simplified approach provides all the necessary functionality while maintaining clean, maintainable code that's easy to understand and modify.
