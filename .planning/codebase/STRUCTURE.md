# STRUCTURE.md — Codebase Structure

## Directory Tree (annotated)

```
fake-new-detection/
├── src/                          # Core source code
│   ├── main.py                   # Entry point — sets OPENSSL_CONF, launches crawl
│   ├── OVERVIEW.md               # Detailed system documentation
│   ├── openssl.cnf               # Legacy SSL config for old Vietnamese sites
│   │
│   ├── crawler/                  # Async web crawler engine
│   │   ├── base_crawler.py       # Abstract BaseCrawler (fetch, selectors, retry)
│   │   ├── crawler_factory.py    # CrawlerFactory + CrawlJournal (resume logic)
│   │   ├── crawl_result.py       # CrawlResult dataclass
│   │   ├── output_formats.py     # OutputFormatter → JSON schema
│   │   ├── typings.py            # SelectorType etc.
│   │   └── news/
│   │       ├── real/             # 9 site-specific crawlers (VnExpress, DanTri…)
│   │       └── fake/             # Fake-news source crawlers
│   │
│   ├── preprocessing/            # Data preprocessing pipeline
│   │   ├── text_preprocessing.py      # Vietnamese text clean → PhoBERT tokenize
│   │   ├── image_preprocessing.py     # Image resize, normalize → tensors
│   │   ├── combined_preprocessing.py  # Unified text+image pipeline
│   │   ├── data_utils.py              # Dataset utilities, data loading helpers
│   │   ├── evidence_retrieval.py      # Evidence/claim pairing logic
│   │   ├── example_preprocessing.py   # Usage examples
│   │   └── coolant/                   # COOLANT-specific preprocessing
│   │       ├── pair_extractor.py      # Extract (claim, evidence) pairs
│   │       ├── pair_dataset.py        # PyTorch Dataset for pairs
│   │       └── training_utils.py      # DataLoader helpers for training
│   │
│   ├── models/                   # Neural network models
│   │   ├── base.py               # BaseModel, MultimodalModel, FastCNN, ContrastiveLoss
│   │   ├── coolant.py            # COOLANT model (EncodingPart, VAE, CrossModule…)
│   │   ├── coolant_official.py   # Paper-faithful COOLANT variant
│   │   ├── resnet_coolant.py     # ResNet-backbone COOLANT
│   │   ├── clip_model.py         # CLIP contrastive model
│   │   ├── senet.py              # SEBlock, SENetwork, SEAttentionModule
│   │   ├── config.py             # Dataclass configs for all models
│   │   ├── factory.py            # ModelFactory, ModelBuilder
│   │   ├── __init__.py           # Public API, AVAILABLE_MODELS registry
│   │   └── README.md             # Model documentation
│   │
│   ├── helpers/                  # Shared utilities
│   │   ├── httpx_client.py       # Async HTTP client (retry, User-Agent, SSL)
│   │   ├── file_handler/         # File I/O abstractions
│   │   ├── json_helper.py        # JSON read/write utilities
│   │   ├── string_handle.py      # String manipulation helpers
│   │   ├── logger.py             # Loguru logger singleton
│   │   ├── paths.py              # get_data_root() — DATA_ROOT from env
│   │   ├── google_drive_uploader.py  # Google Drive upload logic
│   │   └── legacy_tool_handler.py    # Legacy compatibility
│   │
│   ├── parser/                   # (additional parsing utilities)
│   ├── utils/                    # device.py (get_device), misc utils
│   ├── exceptions/               # Custom exception classes
│   └── typings/                  # Shared TypedDict/type alias definitions
│
├── tests/                        # pytest test suite
│   ├── conftest.py               # (empty — shared fixtures TBD)
│   ├── crawler/test_simple_crawler.py
│   ├── helpers/test_json_helper.py, test_string_handle.py
│   └── processing/coolant/test_pair_extractor.py
│
├── notebooks/                    # Operational Jupyter notebooks
│   ├── pipeline/                 # 01_data_crawling.ipynb, 02_preprocessing.ipynb…
│   └── google_drive_upload.ipynb
│
├── examples/                     # Standalone usage examples
│   ├── simple_pipeline.py
│   └── train_coolant_official.py
│
├── vastai/                       # Remote GPU training utilities
├── diagrams/                     # draw.io architecture diagrams
├── docs/                         # Extended docs & papers
├── archive/                      # Archived code & MLflow runs
├── data_archived_20260607/       # Archived crawl status caches
│
├── pyproject.toml                # Project metadata + uv deps + pytest config
├── requirements.txt              # pip-compatible requirements mirror
├── environment.yml               # Conda environment definition
├── config.json                   # Runtime config
├── .env.mac / .env.*.example     # Environment-specific secrets
└── .gitignore
```

## Key Entry Points
| Purpose | File |
|---------|------|
| Run crawler | `python src/main.py` |
| Train COOLANT | `examples/train_coolant_official.py` |
| Preprocessing pipeline | `notebooks/pipeline/02_preprocessing.ipynb` |
| Run tests | `pytest` (from project root) |
| Upload to Drive | `notebooks/google_drive_upload.ipynb` |
