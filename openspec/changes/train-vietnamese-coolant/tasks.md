## 1. Data Preparation

-   [x] 1.1 Verify HDF5 dataset structure (text_features, image_features, labels) for Vietnamese data.
-   [x] 1.2 Implement or refine the `HDF5Dataset` class in `src/processing/hdf5_dataset.py` for on-demand loading.
-   [x] 1.3 Add a factory function for HDF5-based DataLoaders with split management.

## 2. Model Refinement

-   [x] 2.1 Create a `ResNetCOOLANT` wrapper to adapt COOLANT for 2048-dim ResNet50 and 768-dim BERT features.
-   [x] 2.2 Implement the layer patching logic to adjust `shared_image` and `fast_cnn` input dimensions.

## 3. Training Loop Implementation

-   [x] 3.1 Setup three distinct optimizers (Similarity, CLIP, Detection) with appropriate learning rates.
-   [x] 3.2 Implement the multi-task `run_epoch` function following the official sequential task architecture.
-   [x] 3.3 Integrate multi-component loss calculation including CosineEmbeddingLoss and KLDivLoss.

## 4. Tracking & Evaluation

-   [x] 4.1 Integrate MLflow logging for all hyperparameters and per-epoch metrics.
-   [x] 4.2 Implement best-model checkpointing and log artifacts to MLflow.
-   [x] 4.3 Develop a final evaluation script that generates a confusion matrix and detailed classification report.

## 5. Final Integration

-   [x] 5.1 Consolidate the training logic into a production-ready notebook `notebooks/train_vietnamese_coolant.ipynb`.
-   [ ] 5.2 Conduct a full training run to verify end-to-end performance and documentation accuracy.
