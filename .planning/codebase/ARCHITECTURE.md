# ARCHITECTURE.md — System Architecture

## System Overview
Vietnamese multimodal fake news detection — end-to-end pipeline from URL crawling to binary classification (Real/Fake).

## Four-Stage Pipeline

```
Stage 1: Acquisition
  Input: ViFactCheck URLs (JSON)
  → CrawlerFactory (domain routing)
  → Site-specific BaseCrawler subclass (async httpx)
  → HTML parse → extract title, content text, image URLs
  → Download & convert images → JPG
  Output: JSON data files + raw image files

Stage 2: Preprocessing
  Text:  regex clean → underthesea normalize → PhoBERT tokenize → embeddings
  Image: resize 224×224 → normalize (mean/std) → tensors
  Output: PyTorch tensors ready for model

Stage 3: Modeling (COOLANT)
  → EncodingPart: FastCNN (text) + Linear layers (image) → shared dim
  → SimilarityModule: cross-modal similarity matrix
  → AmbiguityLearning (VAE): KL divergence between text & image distributions
  → UnimodalDetection: per-modality unimodal predictions
  → CrossModule4Batch: cross-modal correlation tensor
  → SEAttentionModule: squeeze-and-excitation feature reweighting
  → DetectionModule: final binary classifier
  Loss = α·classification + β·contrastive + γ·similarity

Stage 4: Storage
  → Processed JSON datasets
  → Model checkpoints (.pth / safetensors)
  → MLflow experiment logs
  → (Optional) Google Drive upload
```

## Model Architecture Detail: COOLANT

```
Text input (PhoBERT embeddings, dim=200)
    └─ FastCNN (multi-kernel: 1,2,4,8) → 128-dim
Image input (ResNet features, dim=512)
    └─ Linear projection → 128-dim

    ↓ EncodingPart (shared_dim=128)

SimilarityModule
    └─ cross-modal dot-product similarity

AmbiguityLearning (VAE)
    ├─ Encoder: text/image → μ, σ → Normal distribution
    ├─ Reparameterization trick → z samples
    └─ Symmetric KL divergence (skl) = ambiguity score

UnimodalDetection
    ├─ Text-only classifier branch
    └─ Image-only classifier branch

CrossModule4Batch
    └─ Attention over cross-modal correlation

SEAttentionModule (Squeeze-and-Excitation)
    └─ Channel-wise feature reweighting

DetectionModule
    └─ Fused representation → binary output (Real=0, Fake=1)
```

Additional model variants:
- `coolant_official.py` — closer to paper's original implementation
- `resnet_coolant.py` — ResNet backbone variant
- `clip_model.py` — CLIP-based contrastive model

## Crawler Architecture

```
CrawlerFactory
  ├─ CRAWLER_MAPPING: {domain → CrawlerClass}
  ├─ CrawlJournal: resume state (completed + failed URL caches)
  └─ async crawl loop (semaphore-limited concurrency)

BaseCrawler (abstract)
  ├─ title_selector (CSS/XPath property)
  ├─ content_selector
  ├─ image_selector
  └─ fetch() → CrawlResult

Site Crawlers (inherit BaseCrawler):
  news/real/VnExpressCrawler.py
  news/real/DanTriCrawler.py
  ... (9 real, 1 fake directory)

OutputFormatter
  └─ formats CrawlResult → JSON output schema
```

## Key Architectural Decisions
1. **Factory Pattern** for crawlers — new sources added without modifying core code
2. **Resume capability** — `CrawlJournal` persists completed/failed URLs; interrupted crawls resume from where they left off
3. **SSL override at process level** — `OPENSSL_CONF` env var set in `main.py` before any network calls
4. **Ambiguity as a signal** — VAE-derived KL divergence explicitly models text-image semantic gap (core COOLANT innovation)
5. **Composite loss** — three-part loss (classification + contrastive + similarity) trains multiple objectives simultaneously
