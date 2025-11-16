#!/usr/bin/env python3
"""
Initialize Vietnamese Text Preprocessing
Usage example for Vietnamese fake news detection
"""

import sys
import os
import json
import numpy as np

# Add src to path
sys.path.append('./src')

from preprocessing.text_preprocessing import TextPreprocessor, preprocess_vietnamese_data

def init_preprocessing():
    """Initialize Vietnamese text preprocessing with sample data"""
    
    print("🇻🇳 Initializing Vietnamese Text Preprocessing for COOLANT")
    print("=" * 60)
    
    # Initialize preprocessor for Vietnamese
    preprocessor = TextPreprocessor(
        model_name="vinai/phobert-base",  # Vietnamese BERT model
        language="vi",
        max_length=512,
        device="cuda" if os.system("nvidia-smi") == 0 else "cpu"
    )
    
    print(f"✓ Preprocessor initialized with model: {preprocessor.model_name}")
    print(f"✓ Device: {preprocessor.device}")
    print(f"✓ Max sequence length: {preprocessor.max_length}")
    
    # Sample Vietnamese texts for testing
    sample_texts = [
        "Tin tức Việt Nam hôm nay: Chính phủ ban hành chính sách mới về kinh tế.",
        "Cảnh báo: Tin giả về dịch bệnh COVID-19 đang lan truyền trên mạng xã hội.",
        "Khoa học công nghệ Việt Nam đạt nhiều thành tựu quan trọng trong năm 2023.",
        "BREAKING: Phát hiện thuốc chữa bách bệnh - các chuyên gia cảnh báo tin giả."
    ]
    
    sample_labels = [0, 1, 0, 1]  # 0: real news, 1: fake news
    
    print(f"\n📝 Processing {len(sample_texts)} sample Vietnamese texts...")
    
    # Test text cleaning
    print("\nOriginal texts:")
    for i, text in enumerate(sample_texts):
        print(f"  {i+1}. {text}")
    
    print("\nCleaned texts:")
    for i, text in enumerate(sample_texts):
        cleaned = preprocessor.clean_text(text)
        print(f"  {i+1}. {cleaned}")
    
    # Extract features
    print("\n🔧 Extracting BERT features...")
    
    # Option 1: BERT features (pooled output)
    bert_features = preprocessor.extract_bert_features(sample_texts)
    print(f"✓ BERT features shape: {bert_features.shape}")
    
    # Option 2: Token embeddings (for FastCNN)
    token_embeddings = preprocessor.extract_token_embeddings(sample_texts)
    print(f"✓ Token embeddings shape: {token_embeddings.shape}")
    
    # Save processed data
    print("\n💾 Saving processed data...")
    
    # Create output directory
    os.makedirs("./processed_data", exist_ok=True)
    
    # Save BERT features
    preprocessor.save_preprocessed_data(
        bert_features, 
        np.array(sample_labels), 
        "./processed_data/vietnamese_bert_features.pkl"
    )
    
    # Save token embeddings
    preprocessor.save_preprocessed_data(
        token_embeddings, 
        np.array(sample_labels), 
        "./processed_data/vietnamese_token_embeddings.pkl"
    )
    
    print("\n🎉 Vietnamese text preprocessing initialization completed!")
    print("\n📁 Files created:")
    print("  - ./processed_data/vietnamese_bert_features.pkl")
    print("  - ./processed_data/vietnamese_token_embeddings.pkl")
    
    return preprocessor

def process_custom_dataset(data_path: str):
    """Process custom Vietnamese dataset from JSON file"""
    
    print(f"\n📂 Processing dataset from: {data_path}")
    
    try:
        # Load Vietnamese dataset
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Extract texts and labels (adjust based on your data structure)
        if isinstance(raw_data, list):
            texts = [item.get('text', item.get('content', '')) for item in raw_data]
            labels = [item.get('label', item.get('is_fake', 0)) for item in raw_data]
        elif isinstance(raw_data, dict):
            texts = raw_data.get('texts', [])
            labels = raw_data.get('labels', [])
        else:
            raise ValueError("Unsupported data format")
        
        print(f"✓ Loaded {len(texts)} texts and {len(labels)} labels")
        
        # Initialize preprocessor
        preprocessor = TextPreprocessor(
            model_name="vinai/phobert-base",
            language="vi"
        )
        
        # Process dataset
        features, processed_labels = preprocessor.preprocess_dataset(
            texts, labels,
            save_path="./processed_data/custom_vietnamese_dataset.pkl",
            extract_type="token_embeddings"  # FastCNN compatible
        )
        
        print(f"✓ Processed features shape: {features.shape}")
        print(f"✓ Saved to: ./processed_data/custom_vietnamese_dataset.pkl")
        
        return features, processed_labels
        
    except Exception as e:
        print(f"❌ Error processing dataset: {e}")
        return None, None

def main():
    """Main function to run initialization"""
    
    # Create output directory
    os.makedirs("./processed_data", exist_ok=True)
    
    # Run initialization
    preprocessor = init_preprocessing()
    
    # Optional: Process custom dataset
    custom_data_path = "./src/data/json/news_data_vifactcheck_dev.json"
    if os.path.exists(custom_data_path):
        print("\n" + "="*60)
        print("🎯 Processing your Vietnamese dataset...")
        features, labels = process_custom_dataset(custom_data_path)
        
        if features is not None:
            print(f"\n🚀 Your Vietnamese dataset is ready for COOLANT training!")
            print(f"   Features shape: {features.shape}")
            print(f"   Labels shape: {labels.shape}")
    else:
        print(f"\n⚠️  Custom dataset not found at: {custom_data_path}")
        print("   You can manually process your dataset using the process_custom_dataset() function")
    
    print("\n🎉 Vietnamese preprocessing setup completed!")

if __name__ == "__main__":
    main()
