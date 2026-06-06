# COOLANT Training Workflow

Run these notebooks in order on a machine with GPU.

## Steps

### Step 0: Verify Labels (`0_verify_labels.ipynb`)
- Runs `recover_labels.py` to inject ViFactCheck labels into crawled data
- Uses `binary_nei_as_real` strategy (NEI treated as Real for better balance)
- **Run once** (or after re-crawling)

### Step 1: Preprocess (`1_preprocess.ipynb`)
- Extracts PhoBERT text embeddings `[512, 768]` and ResNet50 image features `[2048]`
- Saves separate HDF5 per split: `vifactcheck_{train,dev,test}.h5`
- Uses zero-image placeholders for missing images (not random noise)
- **Run once** (or after changing preprocessing config)
- Takes ~2-4 hours depending on hardware

### Step 2: Train (choose one or more)

#### 2a: Simultaneous Training (`2a_train_simultaneous.ipynb`) - Paper-faithful
- Official COOLANT approach: all 3 modules trained every epoch
- 3 separate AdamW optimizers (Similarity, CLIP, Detection)
- Class-weighted loss + weighted sampling for imbalance
- Early stopping on macro F1

#### 2b: Phased Training (`2b_train_phased.ipynb`) - Experimental
- Phase 1: CLIP pre-training (10 epochs, self-supervised)
- Phase 2: Similarity pre-training (10 epochs, CLIP frozen)
- Phase 3: Detection training (20 epochs, CLIP+Sim frozen)
- Phase 4: Joint fine-tuning (10 epochs, 0.1x LR)
- Same class balancing as 2a

#### 2c: AdaBelief Training (`2c_train_adabelief.ipynb`) - Experimental
- Same simultaneous approach as 2a but with AdaBelief optimizer
- AdaBelief adapts step sizes by belief in observed gradients (NeurIPS 2020)
- `eps=1e-16`, `rectify=True`, `weight_decouple=True`
- Install: `pip install adabelief-pytorch`
- Compare macro F1 against 2a to evaluate optimizer impact

## Data Flow

```
ViFactCheck HuggingFace
        |
  recover_labels.py (Step 0)
        |
  labeled_nei_as_real/*.json
    train: 341 real + 4168 fake
    dev:   311 real + 323 fake (balanced!)
    test:  496 real + 786 fake
        |
  PhoBERT + ResNet50 (Step 1)
        |
  vifactcheck_{train,dev,test}.h5
        |
  COOLANT training (Step 2a, 2b, or 2c)
        |
  checkpoints/ or checkpoints_phased/ or checkpoints_adabelief/
```

## Expected Results

- Training loss should decrease across epochs
- Both classes should appear in confusion matrix predictions
- Macro F1 should exceed 0.50 (above random baseline)
- Dev set is ~49/51 balanced, so accuracy is meaningful there
