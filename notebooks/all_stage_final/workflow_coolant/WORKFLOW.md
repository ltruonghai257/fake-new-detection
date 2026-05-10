# COOLANT Image-Caption Training Workflow

Paper-faithful COOLANT training using image-caption pairs.

## How COOLANT Labels Data (arXiv:2302.14057)

- **Real**: matched pair (caption_i, image_i) from the same article
- **Fake**: unmatched pair (caption_i, image_j) from different articles
- Labels are created **dynamically per batch** — no external labels needed
- **Balanced by construction** — 50% Real, 50% Fake in every batch

## Steps

### Step 0: Extract Pairs (`0_extract_pairs.ipynb`)
- Extracts (image_path, caption) pairs from crawled JSON
- Filters: caption must be non-empty, image must exist on disk
- Output: `./pairs/pairs_{train,dev,test}.json`

### Step 1: Preprocess (`1_preprocess.ipynb`)
- Caption -> PhoBERT embeddings [128, 768] (short captions, not full articles)
- Image -> ResNet50 features [2048]
- Output: `./processed/coolant_{train,dev,test}.h5`

### Step 2: Train

#### 2a: Simultaneous (`2a_train_simultaneous.ipynb`)
- Paper-faithful: all 3 modules per epoch
- No class weights needed (balanced pairs)

#### 2b: Phased (`2b_train_phased.ipynb`)
- Phase 1: CLIP (10 ep) -> Phase 2: Similarity (10 ep) -> Phase 3: Detection (20 ep) -> Phase 4: Joint (10 ep)

## Reusable Modules

All logic lives in `src/processing/coolant/`:
- `pair_extractor.py` — Extract pairs from JSON
- `pair_dataset.py` — PyTorch Dataset + DataLoader
- `training_utils.py` — Matched/unmatched pair construction

## Data Flow

```
Crawled JSON (articles with images + captions)
     |
  0_extract_pairs.ipynb (PairExtractor)
     |
  pairs_{train,dev,test}.json
     |
  1_preprocess.ipynb (PhoBERT + ResNet50)
     |
  coolant_{train,dev,test}.h5
     |
  2a or 2b training (CoolantPairDataset + make_coolant_pairs)
     |
  checkpoints_coolant/ or checkpoints_coolant_phased/
```
