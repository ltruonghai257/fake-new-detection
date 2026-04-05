# Tasks: Train Vietnamese COOLANT Misinformation Detection

## Phase 1: Data Preparation (1 day)

### Task 1.1: Verify HDF5 Dataset
**Status**: ready  
**File**: `notebooks/4_train_model.ipynb` cell 5  
**Description**: Ensure HDF5 file exists with correct structure

**Steps**:
1. Check if `/content/drive/MyDrive/Thesis_Final/fake-new-detection/notebooks/processed_data/crawled/dataset.h5` exists
2. If not, run conversion from NPZ
3. Verify dataset attributes:
   - n_samples = 4814
   - train_size ≈ 3851 (80%)
   - val_size ≈ 481 (10%)
   - test_size ≈ 482 (10%)
   - text_shape = (4814, 512, 768)
   - image_shape = (4814, 2048)

**Validation**:
```python
with h5py.File(HDF5_PATH, 'r') as f:
    assert f.attrs['n_samples'] == 4814
    assert f['text_features'].shape == (4814, 512, 768)
```

---

### Task 1.2: Apply Label Guard
**Status**: ready  
**File**: `notebooks/4_train_model.ipynb` cell 6  
**Description**: Ensure binary classification labels exist

**Steps**:
1. Load labels from HDF5
2. Check unique values
3. If single class (all 0s), flip 30% to 1
4. Save modified labels back to HDF5
5. Verify distribution: ~70% real, ~30% fake

**Expected Output**:
```
Original: {0: 4814}
After synthesis: {0: 3370, 1: 1444}
```

---

### Task 1.3: Create HDF5 DataLoaders
**Status**: ready  
**File**: `notebooks/4_train_model.ipynb` cell 9  
**Description**: Initialize memory-efficient data loaders

**Steps**:
1. Import `create_hdf5_dataloaders` from `src.processing.hdf5_dataset`
2. Set `SEED = 42`, `BATCH_SIZE = 32`
3. Create train/val/test loaders
4. Verify batch shapes:
   - text: (32, 512, 768)
   - image: (32, 2048)
   - labels: (32,)

**Validation**:
```python
batch = next(iter(train_loader))
assert batch[0].shape == (32, 512, 768)  # text
assert batch[1].shape == (32, 2048)      # image
assert batch[2].shape == (32,)           # labels
```

---

## Phase 2: Model Setup (2 days)

### Task 2.1: Initialize ResNetCOOLANT
**Status**: ready  
**File**: `notebooks/4_train_model.ipynb` cell 11  
**Description**: Create adapted COOLANT model

**Steps**:
1. Import COOLANT_Official from `src.models.coolant_official`
2. Define ResNetCOOLANT subclass with:
   - `encode_text()` method
   - `encode_image()` method
   - `fuse_modalities()` method
3. Instantiate with CONFIG dict
4. Move model to DEVICE (cuda)

**Expected Output**:
```
Trainable parameters: ~2-3 million
Model ready ✓
```

---

### Task 2.2: Apply Layer Patches
**Status**: ready  
**File**: `notebooks/4_train_model.ipynb` cell 11 (patches)  
**Description**: Adapt layers for ResNet50 and BERT dimensions

**Steps**:

**Patch A**: EncodingPart.shared_image (512 → 2048)
- Apply to `similarity_module.encoding`
- Apply to `detection_module.encoding`
- Apply to `detection_module.ambiguity_module.encoding`

**Patch B**: CLIP.image_projection (512 → 2048)
- Replace first Linear layer

**Patch C**: FastCNN input_dim (200 → 768)
- Replace fast_cnn in all shared_text_encoding modules

**Validation**:
```python
# Test forward pass with dummy input
dummy_text = torch.randn(2, 512, 768).to(DEVICE)
dummy_image = torch.randn(2, 2048).to(DEVICE)
output = model(dummy_text, dummy_image)
assert output is not None
```

---

### Task 2.3: Setup Optimizers and Schedulers
**Status**: ready  
**File**: `notebooks/4_train_model.ipynb` cell 12  
**Description**: Configure training optimization

**Steps**:
1. Create optimizers:
   - `optim_sim`: Adam for similarity_module (lr=1e-3)
   - `optim_clip`: AdamW for clip_module (lr=1e-3, wd=5e-4)
   - `optim_det`: Adam for detection_module (lr=1e-3)
2. Create StepLR schedulers (step_size=10, gamma=0.5)
3. Define loss functions:
   - `loss_cos`: CosineEmbeddingLoss(margin=0.2)
   - `loss_ce`: CrossEntropyLoss()
   - `loss_kl`: KLDivLoss(reduction='batchmean')
4. Create SAVE_DIR for checkpoints

**Validation**:
```
Optimizers ready ✓
```

---

## Phase 3: Training (3-5 days)

### Task 3.1: Implement Training Loop
**Status**: ready  
**File**: `notebooks/4_train_model.ipynb` cells 14-16  
**Description**: Complete epoch training with all three tasks

**Steps**:
1. Implement `make_sim_pairs()`: create matched/unmatched pairs
2. Implement `soft_xe()`: soft cross-entropy for knowledge distillation
3. Implement `run_epoch()`:
   - Loop through batches
   - Task 1a: Cosine similarity loss
   - Task 1b: CLIP contrastive loss
   - Task 2: Detection loss
   - Accumulate metrics (loss, accuracy)
4. Run for NUM_EPOCHS = 30
5. Save best model based on val_acc

**Expected Output**:
```
Epoch [01/30]  train loss=0.6931 acc=0.6500  |  val loss=0.6234 acc=0.7000
Epoch [02/30]  train loss=0.6123 acc=0.7000  |  val loss=0.5834 acc=0.7200
...
  ✓ Best val_acc=0.8234 saved.
```

**Validation**:
- Training loss decreases
- Validation accuracy increases
- No NaN values in losses

---

### Task 3.2: Monitor Training Progress
**Status**: ready  
**File**: `notebooks/4_train_model.ipynb` cell 18  
**Description**: Track and visualize training metrics

**Steps**:
1. Store history: train_loss, train_acc, val_loss, val_acc
2. Save history to JSON
3. Plot training curves:
   - Loss (train vs val)
   - Accuracy (train vs val)
4. Save plots as PNG

**Expected Output**:
```
Training done. Best val acc: 0.8234
History saved.
```

**Validation**:
- `training_history.json` exists with 30 epochs
- `training_curves.png` shows convergence

---

## Phase 4: Evaluation (1 day)

### Task 4.1: Final Test Evaluation
**Status**: ready  
**File**: `notebooks/4_train_model.ipynb` cell 20  
**Description**: Evaluate on held-out test set

**Steps**:
1. Load best model checkpoint
2. Run `run_epoch(test_loader, train=False)`
3. Calculate metrics:
   - Loss
   - Accuracy
   - Precision (weighted)
   - Recall (weighted)
   - F1-Score (weighted)
4. Generate classification report (per-class metrics)
5. Save results to JSON

**Success Criteria**:
| Metric | Target | Minimum |
|--------|--------|---------|
| Accuracy | >80% | >75% |
| F1-Score | >0.78 | >0.70 |

**Expected Output**:
```
=== Test Results ===
Loss      : 0.5234
Accuracy  : 0.8234
Precision : 0.8200
Recall    : 0.8234
F1-Score  : 0.8215

              precision    recall  f1-score   support
Real          0.85        0.90      0.87       337
Fake          0.75        0.68      0.71       145
Test results saved.
```

---

### Task 4.2: Generate Confusion Matrix
**Status**: ready  
**File**: `notebooks/4_train_model.ipynb` cell 21  
**Description**: Visualize prediction errors

**Steps**:
1. Get predictions from test set
2. Create confusion matrix
3. Plot heatmap with Seaborn
4. Save as PNG

**Expected Output**:
- `confusion_matrix.png` with 2×2 matrix
- Shows true positives, false positives, etc.

---

### Task 4.3: Save Final Model
**Status**: ready  
**File**: `notebooks/4_train_model.ipynb` cell 22  
**Description**: Export complete model package

**Steps**:
1. Save comprehensive checkpoint:
   - model_state_dict
   - config
   - best_val_acc
   - test_results
   - history
2. Save as `coolant_resnet50_final.pth`

**Expected Output**:
```
Saved → ./training/checkpoints/coolant_resnet50_final.pth
  Best val acc : 0.8234
  Test acc     : 0.8234
  Test F1      : 0.8215
```

---

## Task Summary

| # | Task | Phase | Est. Time |
|---|------|-------|-----------|
| 1.1 | Verify HDF5 Dataset | Data Prep | 2h |
| 1.2 | Apply Label Guard | Data Prep | 1h |
| 1.3 | Create DataLoaders | Data Prep | 2h |
| 2.1 | Initialize Model | Setup | 2h |
| 2.2 | Apply Patches | Setup | 4h |
| 2.3 | Setup Optimizers | Setup | 1h |
| 3.1 | Training Loop | Training | 3-5 days |
| 3.2 | Monitor Progress | Training | 2h |
| 4.1 | Test Evaluation | Evaluation | 2h |
| 4.2 | Confusion Matrix | Evaluation | 1h |
| 4.3 | Save Model | Evaluation | 1h |

**Total Estimated Time**: 6-8 days

---

## Dependencies

### Blockers
- Task 1.1 → Task 1.2 → Task 1.3 (sequential)
- Task 1.3 → Task 2.1 → Task 2.2 → Task 2.3 (sequential)
- Task 2.3 → Task 3.1 → Task 3.2 (sequential)
- Task 3.2 → Task 4.1 → Task 4.2 → Task 4.3 (sequential)

### Parallelizable
- Task 4.2 and Task 4.3 can run together after 4.1

---

**Status**: All tasks ready for implementation
**Next Step**: Run `/opsx:apply` or `skill: openspec-apply-change` to begin implementation
