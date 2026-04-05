## Context

The project currently has a basic implementation of the COOLANT model, but it needs to be robustly trained on a Vietnamese multimodal dataset. The dataset consists of Vietnamese news articles with corresponding images. Text features are extracted using a Vietnamese-BERT model, and image features are extracted using ResNet50.

Current constraints:
- Image features are 2048-dimensional (ResNet50).
- Text features are 768-dimensional (BERT).
- Large datasets require memory-efficient loading (HDF5).
- Training requires tracking multiple loss components (Classification, CLIP, Similarity, Ambiguity).

## Goals / Non-Goals

**Goals:**
- Adapt the COOLANT model to handle 2048-dim image features and 768-dim text features.
- Implement an HDF5-based data loader for efficient multimodal data access.
- Implement a multi-task training loop following the official COOLANT repository's architecture (Similarity Learning, CLIP Contrastive, and Detection with Ambiguity Learning).
- Integrate MLflow for comprehensive experiment tracking and model versioning.
- Provide a clear and reproducible training script/notebook for both local and Colab environments.

**Non-Goals:**
- Training the feature extraction models (ResNet50, BERT) from scratch; we use pre-extracted features.
- Implementing a real-time inference server; the focus is on the training and evaluation pipeline.

## Decisions

### 1. Model Architecture Adaptation
**Decision**: Use a "patching" approach or a dedicated subclass to adapt the internal layers of `COOLANT_Official` for 2048-dim image inputs.
**Rationale**: The base COOLANT implementation often assumes 512-dim image features. Using a subclass `ResNetCOOLANT` allows us to override the input layers without modifying the core model logic, ensuring maintainability.
**Alternatives**: Modifying the base `COOLANT_Official` class directly, which would make it harder to pull updates from the official repository.

### 2. Data Loading Strategy
**Decision**: Use `HDF5Dataset` with `torch.utils.data.DataLoader`.
**Rationale**: HDF5 allows for random access to large datasets without loading them entirely into memory. Setting `num_workers=0` is necessary for HDF5 to avoid issues with multiple processes opening the same file, or carefully managing file handles if parallel loading is required.
**Alternatives**: Using standard PyTorch `Dataset` with individual files, which leads to slow I/O when dealing with thousands of small files.

### 3. Training Loop Structure
**Decision**: Implement three separate optimizers for Similarity, CLIP, and Detection tasks, as seen in the official COOLANT repository.
**Rationale**: Each task has different learning objectives and convergence rates. Separate optimizers allow for finer control over learning rates and weight decays for each module.
**Alternatives**: Using a single optimizer for all parameters, which might lead to sub-optimal convergence for some tasks.

### 4. Experiment Tracking
**Decision**: Use MLflow.
**Rationale**: MLflow is industry-standard for tracking experiments, logging metrics, and saving model artifacts. It integrates well with PyTorch and provides a clean UI for comparison.
**Alternatives**: Using TensorBoard or simple JSON logging. MLflow is preferred for its superior artifact management and experiment organization.

## Risks / Trade-offs

- **[Risk]** HDF5 I/O bottleneck → **[Mitigation]** Use fast SSD storage and consider chunking the HDF5 file appropriately.
- **[Risk]** Overfitting on the limited Vietnamese dataset → **[Mitigation]** Implement early stopping and monitor validation metrics closely via MLflow.
- **[Risk]** Discrepancy between paper and implementation → **[Mitigation]** Follow the official repository's implementation as it is the "ground truth" for reproducibility.
