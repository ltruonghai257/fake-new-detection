## Why

Existing fake news detection models often struggle with the nuances of Vietnamese media and the complex relationship between text and imagery in this context. Training the COOLANT (Cross-modal Contrastive Learning for Multimodal Fake News Detection) model specifically on Vietnamese datasets will provide a robust, state-of-the-art solution that leverages both textual and visual evidence for more accurate misinformation detection.

## What Changes

- **Training Pipeline**: Implement a comprehensive training and evaluation pipeline tailored for the Vietnamese news dataset.
- **Model Adaptation**: Configure the COOLANT architecture to support ResNet50-extracted image features (2048-dim) and Vietnamese BERT-based text features (768-dim).
- **Data Management**: Utilize memory-efficient HDF5-based data loading to handle large multimodal datasets without exceeding RAM limits.
- **Experiment Tracking**: Integrate MLflow to monitor training progress, log hyperparameters, and track performance metrics across different runs.
- **Checkpointing**: Implement a structured saving mechanism for the best-performing models and training history.

## Capabilities

### New Capabilities
- `vietnamese-coolant-training`: Core training orchestration including multi-task loss computation (similarity, CLIP, and detection) specifically optimized for Vietnamese news patterns.
- `hdf5-multimodal-loading`: An efficient data provider that reads pre-extracted text and image features from HDF5 files on-demand.

### Modified Capabilities
- None

## Impact

- **src/models/**: Addition or refinement of the COOLANT model implementation to ensure compatibility with 2048-dim image features.
- **src/processing/**: Enhancement of data loaders to support HDF5 and Vietnamese-specific preprocessing.
- **training/**: Creation of a dedicated training directory for checkpoints, logs, and MLflow artifacts.
- **notebooks/**: Production-ready training notebooks for Google Colab and local execution.
