## ADDED Requirements

### Requirement: Memory-efficient Multimodal Loading
The system SHALL implement a data provider that reads multimodal features (BERT text and ResNet50 image features) from HDF5 files on-demand during training and evaluation.

#### Scenario: On-demand batch retrieval
- **WHEN** a DataLoader requests a batch of samples
- **THEN** the system SHALL ONLY read the specific indices from the `text_features` and `image_features` datasets in the HDF5 file

### Requirement: HDF5 Feature Schema Support
The system SHALL be compatible with HDF5 files containing at least three datasets: `text_features` (shape [N, 30, 768] or [N, 768]), `image_features` (shape [N, 2048]), and `labels` (shape [N]).

#### Scenario: Metadata validation
- **WHEN** the HDF5 dataset is initialized
- **THEN** it SHALL verify that all required datasets exist and have consistent lengths

### Requirement: Cross-validation Split Management
The system SHALL support partitioning the HDF5 data into `train`, `val`, and `test` splits based on pre-defined or dynamically computed index ranges.

#### Scenario: Split-aware data loading
- **WHEN** a `train_loader` is requested
- **THEN** it SHALL only provide samples that belong to the training subset of the HDF5 file
