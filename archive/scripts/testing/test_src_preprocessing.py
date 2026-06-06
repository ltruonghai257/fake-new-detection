#!/usr/bin/env python3
"""
Test script to verify the preprocessing module works correctly from src/
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.append('./src')

def test_src_imports():
    """Test that all module components can be imported from src/"""
    print("Testing imports from src/preprocessing...")
    
    try:
        # Test main classes
        from preprocessing import (
            TextPreprocessor, 
            ImagePreprocessor, 
            CombinedPreprocessor,
            DataManager,
            DatasetSplitter,
            DataValidator
        )
        print("✓ Main classes imported successfully from src/")
        
        # Test convenience functions
        from preprocessing import (
            preprocess_text_dataset,
            preprocess_image_dataset,
            preprocess_multimodal_dataset
        )
        print("✓ Convenience functions imported successfully from src/")
        
        return True
        
    except Exception as e:
        print(f"✗ Import error from src/: {e}")
        return False

def test_src_functionality():
    """Test basic functionality from src/"""
    print("\nTesting basic functionality from src/...")
    
    try:
        from preprocessing import DataManager
        
        # Test data manager
        manager = DataManager("./test_src_output")
        test_data = {'test': np.array([1, 2, 3])}
        
        # Test save/load
        path = manager.save_pickle(test_data, "test.pkl")
        loaded = manager.load_pickle("test.pkl")
        
        if np.array_equal(test_data['test'], loaded['test']):
            print("✓ Data manager save/load works correctly from src/")
        else:
            print("✗ Data manager save/load failed from src/")
            return False
        
        # Clean up
        os.remove(path)
        os.rmdir("./test_src_output")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test error from src/: {e}")
        return False

def main():
    """Run all tests for src/ preprocessing module"""
    print("COOLANT Preprocessing Module Verification (src/)")
    print("=" * 50)
    
    tests = [
        test_src_imports,
        test_src_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Module in src/ is ready to use.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
