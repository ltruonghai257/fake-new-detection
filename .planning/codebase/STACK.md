# STACK.md — Technology Stack
_Last mapped: 2026-05-08_

## Language & Runtime

- **Python 3.8+** — primary language
- **Conda environment**: `fake_news` (`environment.yml`, prefix `/opt/miniconda3/envs/fake_news`)
- No `requirements.txt` with pinned versions (null-bytes file); dependencies inferred from imports

## Core Frameworks

| Layer | Library | Purpose |
|---|---|---|
| Deep Learning | `torch`, `torchvision` | Model training, tensor ops, ResNet backbones |
| Transformers | `transformers` (HuggingFace) | PhoBERT / ViSoBERT tokenizer + model |
| HTTP client | `httpx` (async) | Resilient web crawling |
| HTML parsing | `beautifulsoup4`, `lxml` | DOM extraction from crawled pages |
| Vietnamese NLP | `underthesea` | Text normalization, word segmentation (optional import) |
| Image processing | `Pillow`, `torchvision.transforms`, `opencv-python` (optional) | Image resize, normalize, JPG conversion |
| Data | `numpy`, `h5py` | Array ops, HDF5 dataset storage |
| Progress | `tqdm` | Crawl/training progress bars |
| Concurrency | `asyncio` | Async crawling semaphore |
| Testing | `pytest` | Unit tests |

## ML Models

| Model | File | Notes |
|---|---|---|
| COOLANT | `src/models/coolant.py` | Cross-modal contrastive learning (research impl) |
| COOLANT_Official | `src/models/coolant_official.py` | Closer-to-paper implementation |
| PatchedCOOLANT / ResNetCOOLANT | `src/models/resnet_coolant.py` | Dimension adapter for arbitrary encoders |
| CLIP | `src/models/clip_model.py` | Contrastive image-language pretraining |
| SENet | `src/models/senet.py` | Squeeze-and-Excitation attention |
| FastCNN | `src/models/base.py` | Multi-kernel 1D CNN for text |

## Pre-trained Models (HuggingFace)

| Key | HF ID | Dim |
|---|---|---|
| `phobert-base` | `vinai/phobert-base` | 768 |
| `phobert-base-v2` | `vinai/phobert-base-v2` | 768 |
| `phobert-large` | `vinai/phobert-large` | 1024 |
| `visobert` | `uitnlp/visobert` | 768 |

## Image Backbones

| Key | Feature Dim |
|---|---|
| `resnet18` | 512 |
| `resnet50` | 2048 |
| `clip-vit-L-14` | 1024 |
| `siglip-base-224` | 768 |

## Configuration

- `config.json` — root-level JSON config
- `openssl.cnf` — legacy SSL override (required for Vietnamese gov sites)
- `src/models/config.py` — typed dataclass configs (`COOLANTConfig`, `CLIPConfig`, `TrainingConfig`, `DataConfig`, etc.)
- `src/utils/device.py` — centralized CUDA/MPS/CPU detection; respects `FORCE_DEVICE` env var

## Third-Party Submodule

- `src/lib/Mocheg/` — embedded git submodule; separate research codebase for retrieval+verification (uses `mlflow`, `wandb`)
