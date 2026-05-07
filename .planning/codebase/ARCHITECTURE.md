# ARCHITECTURE.md — System Architecture
_Last mapped: 2026-05-08_

## Pattern

**Multi-stage research pipeline** with four sequential stages and no central orchestrator. Each stage is independently executable. Factory Pattern used for crawler and model instantiation.

## Stages

```
Stage 1: Acquisition  →  Stage 2: Preprocessing  →  Stage 3: Modeling  →  Stage 4: Storage
  (crawler)                (text + image pipelines)    (COOLANT model)       (JSON + .pth)
```

### Stage 1 — Acquisition (`src/crawler/`)

- Entry: `src/main.py` → `asyncio.run(main())` from `src/test_crawler.py`
- `CrawlerFactory` routes URLs by domain to site-specific `BaseCrawler` subclasses
- `BaseCrawler.arun()` fetches HTML, parses DOM via BeautifulSoup/lxml, downloads images
- Concurrency: `asyncio.Semaphore(max_concurrent=15)` limits parallel requests
- Resume capability: `crawling_status.json` cache — subtract already-crawled from URL list

### Stage 2 — Preprocessing (`src/preprocessing/`, `src/processing/`)

**Text path:**
1. Regex clean (URLs, punctuation, numbers) — `TextPreprocessor`
2. `underthesea.text_normalize()` — Vietnamese accent normalization (graceful fallback if missing)
3. PhoBERT tokenize + embed → `(seq_len, 768)` tensors

**Image path:**
1. Resize to 224×224, normalize (ImageNet mean/std) — `ImagePreprocessor`
2. ResNet50 / CLIP ViT / SigLIP backbone → feature vector (2048 / 1024 / 768 dim)

**Combined:**
- `src/processing/vifactcheck_processor.py` — end-to-end ViFactCheck dataset pipeline
- `src/processing/pytorch_dataset.py` — `FakeNewsDataset(Dataset)` for DataLoader
- `src/processing/hdf5_dataset.py` — HDF5-backed dataset for large preprocessed arrays

### Stage 3 — Modeling (`src/models/`)

**COOLANT architecture:**
```
Text → FastCNN → shared_text_linear → text_shared (128-dim)
Image → shared_image MLP → image_shared (128-dim)
                   ↓
           SimilarityModule (alignment)
                   ↓
           DetectionModule:
             EncodingPart → UnimodalDetection → CrossModule4Batch
             SEAttentionModule (weights text/image/correlation)
             AmbiguityLearning (VAE, symmetric KL divergence)
                   ↓
           Classifier → logits (2-class: real/fake)
```

**Composite loss:**
```python
total_loss = classification_weight * CE_loss
           + contrastive_weight * contrastive_loss (InfoNCE, temp=0.07)
           + similarity_weight * similarity_loss
```

**Model hierarchy:**
- `BaseModel` → `MultimodalModel` → `COOLANT` / `COOLANT_Official`
- `COOLANT_Official` → `PatchedCOOLANT` (alias: `ResNetCOOLANT`) — adapts for arbitrary feature dims
- `ModelFactory` / `ModelBuilder` — factory + builder pattern for instantiation

### Stage 4 — Storage

- Checkpoints: `torch.save({"model_state_dict": ..., "config": ...}, path)`
- Crawl results: JSON files under `data/json/`
- Images: JPG files under `data/jpg/<site>/`

## Key Abstractions

| Abstraction | Location | Role |
|---|---|---|
| `BaseCrawler` | `src/crawler/base_crawler.py` | Abstract contract for all site crawlers |
| `CrawlerFactory` | `src/crawler/crawler_factory.py` | Domain→Crawler routing |
| `MultimodalModel` | `src/models/base.py` | Abstract base: `encode_text`, `encode_image`, `fuse_modalities` |
| `ModelFactory` | `src/models/factory.py` | Model instantiation by name |
| `ImagePreprocessor` | `src/preprocessing/image_preprocessing.py` | Backbone-agnostic image pipeline |
| `TextPreprocessor` | `src/preprocessing/text_preprocessing.py` | PhoBERT/ViSoBERT text pipeline |
| `FakeNewsDataset` | `src/processing/pytorch_dataset.py` | PyTorch Dataset for DataLoader |

## Third-Party Embedded System

`src/lib/Mocheg/` is an embedded git submodule implementing a separate MOCHEG (multimodal claim verification) pipeline with its own retrieval, document, verification, and controllable-generation modules. It is **not integrated** with the main COOLANT pipeline.
