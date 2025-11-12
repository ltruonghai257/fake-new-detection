#!/usr/bin/env python3
"""
Simple ViFactCheck Pipeline Example

Demonstrates the simplified preprocessing and dataloader pipeline.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from processing.simple_preprocess import preprocess_vifactcheck
from processing.simple_dataloader import create_train_val_test_loaders
from processing.text_processor import TextProcessor
from processing.image_processor import ImageProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_preprocessing():
    """Example preprocessing."""
    logger.info("=== Preprocessing Example ===")
    
    # Define paths
    json_path = "src/data/json/news_data_vifactcheck_train.json"
    image_base_dir = "src/data/jpg"
    output_dir = "preprocessed_simple"
    
    try:
        results = preprocess_vifactcheck(
            json_path=json_path,
            image_base_dir=image_base_dir,
            output_dir=output_dir,
            test_size=0.2,
            val_size=0.1
        )
        
        logger.info("Preprocessing completed!")
        logger.info(f"Total samples: {results.get('total_samples', 0)}")
        logger.info(f"Train: {results.get('train_samples', 0)}")
        logger.info(f"Val: {results.get('val_samples', 0)}")
        logger.info(f"Test: {results.get('test_samples', 0)}")
        
        return results
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return None


def example_dataloader(preprocessing_results):
    """Example dataloader creation."""
    logger.info("=== DataLoader Example ===")
    
    if not preprocessing_results:
        logger.error("No preprocessing results available")
        return
    
    # Define label mapping
    label_mapping = {
        'thanh_nien': 0,     # Real news
        'vn_express': 0,     # Real news
        'bao_chinh_phu': 0,  # Real news
        'tuoi_tre': 0,       # Real news
        'dan_tri': 1,        # Fake news (example)
        'bao_tin_tuc': 1,    # Fake news (example)
        'tien_phong': 0      # Real news
    }
    
    try:
        # Create dataloaders
        train_loader, val_loader, test_loader = create_train_val_test_loaders(
            train_path=preprocessing_results['files']['train'],
            val_path=preprocessing_results['files']['val'],
            test_path=preprocessing_results['files']['test'],
            image_base_dir="src/data/jpg",
            batch_size=8,  # Small batch for demo
            num_workers=0,  # Single-threaded for demo
            label_mapping=label_mapping
        )
        
        logger.info("DataLoaders created successfully!")
        
        # Test the dataloaders
        logger.info("\n=== Testing DataLoaders ===")
        
        for name, loader in [("Train", train_loader), ("Val", val_loader), ("Test", test_loader)]:
            logger.info(f"\n{name} Loader:")
            logger.info(f"  Batches: {len(loader)}")
            
            # Get first batch
            try:
                batch = next(iter(loader))
                logger.info(f"  Batch keys: {list(batch.keys())}")
                logger.info(f"  Text features shape: {batch['text_features'].shape}")
                logger.info(f"  Image features shape: {batch['image_features'].shape}")
                logger.info(f"  Labels shape: {batch['labels'].shape}")
                logger.info(f"  Sample text: {batch['texts'][0][:100]}...")
                logger.info(f"  Sample source: {batch['sources'][0]}")
                logger.info(f"  Sample label: {batch['labels'][0].item()}")
            except Exception as e:
                logger.error(f"  Error getting batch: {e}")
        
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        logger.error(f"DataLoader creation failed: {e}")
        return None, None, None


def example_processors():
    """Example individual processor usage."""
    logger.info("=== Individual Processors Example ===")
    
    # Text processor example
    logger.info("\nText Processor:")
    text_processor = TextProcessor(max_length=64)
    
    sample_text = "Chính phủ công bố các biện pháp hỗ trợ người dân trong đại dịch COVID-19."
    text_result = text_processor.process(sample_text)
    
    logger.info(f"  Input: {sample_text}")
    logger.info(f"  Cleaned: {text_result['text']}")
    logger.info(f"  Features shape: {text_result['features'].shape}")
    logger.info(f"  Tokens shape: {text_result['input_ids'].shape}")
    logger.info(f"  Valid: {text_result['valid']}")
    
    # Image processor example
    logger.info("\nImage Processor:")
    image_processor = ImageProcessor(feature_dim=256)
    
    # Create a dummy image for demo
    dummy_image = image_processor.create_dummy_image()
    image_result = image_processor.extract_features(dummy_image)
    
    logger.info(f"  Dummy image size: {dummy_image.size}")
    logger.info(f"  Features shape: {image_result.shape}")


def example_training_preparation(train_loader, val_loader, test_loader):
    """Example training preparation."""
    logger.info("=== Training Preparation Example ===")
    
    if not all([train_loader, val_loader, test_loader]):
        logger.error("DataLoaders not available")
        return
    
    logger.info(f"Training batches per epoch: {len(train_loader)}")
    logger.info(f"Validation batches per epoch: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Show data shapes for model design
    try:
        sample_batch = next(iter(train_loader))
        
        logger.info("\nData shapes for model design:")
        logger.info(f"  Text features: {sample_batch['text_features'].shape}")
        logger.info(f"  Text tokens: {sample_batch['text_tokens'].shape}")
        logger.info(f"  Text mask: {sample_batch['text_mask'].shape}")
        logger.info(f"  Image features: {sample_batch['image_features'].shape}")
        logger.info(f"  Image tensors: {sample_batch['image_tensors'].shape}")
        logger.info(f"  Labels: {sample_batch['labels'].shape}")
        
        # Show label distribution
        import torch
        unique_labels, counts = torch.unique(sample_batch['labels'], return_counts=True)
        label_dist = dict(zip(unique_labels.tolist(), counts.tolist()))
        logger.info(f"  Sample batch label distribution: {label_dist}")
        
    except Exception as e:
        logger.error(f"Error analyzing batch: {e}")


def main():
    """Main pipeline example."""
    logger.info("Starting Simple ViFactCheck Pipeline")
    
    # Step 1: Individual processors
    logger.info("\n" + "="*50)
    example_processors()
    
    # Step 2: Preprocessing
    logger.info("\n" + "="*50)
    preprocessing_results = example_preprocessing()
    
    # Step 3: DataLoader creation
    logger.info("\n" + "="*50)
    train_loader, val_loader, test_loader = example_dataloader(preprocessing_results)
    
    # Step 4: Training preparation
    logger.info("\n" + "="*50)
    example_training_preparation(train_loader, val_loader, test_loader)
    
    logger.info("\n" + "="*50)
    logger.info("Simple Pipeline Example Completed!")
    
    logger.info("\nSummary:")
    logger.info("✓ Separate text and image processors")
    logger.info("✓ Clean preprocessing pipeline")
    logger.info("✓ Simple dataloader implementation")
    logger.info("✓ Ready for model training")


if __name__ == "__main__":
    main()
