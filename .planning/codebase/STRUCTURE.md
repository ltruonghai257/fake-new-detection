# STRUCTURE.md — Directory Layout
_Last mapped: 2026-05-08_

## Root Layout

```
fake-new-detection/
├── src/                    # Main source code
├── tests/                  # Pytest test suite
├── notebooks/              # Jupyter experiment notebooks
├── examples/               # Standalone example scripts
├── docs/                   # Architecture docs (COOLANT, VIFACTCHECK, etc.)
├── diagrams/               # Diagram assets
├── training/               # Training checkpoints (gitignored contents)
├── checkpoints/            # Additional checkpoint dirs
├── processed_data/         # Preprocessed dataset storage
├── placeholder_images/     # Placeholder images for missing data
├── logs/                   # Runtime log files
├── openssl.cnf             # SSL override config (required for crawler)
├── config.json             # Root configuration
├── environment.yml         # Conda environment spec
└── .planning/              # GSD planning documents (this dir)
```

## `src/` Layout

```
src/
├── main.py                 # Entry point: sets OPENSSL_CONF, runs crawler
├── crawler/
│   ├── base_crawler.py     # Abstract BaseCrawler
│   ├── crawler_factory.py  # CrawlerFactory: domain routing + crawl orchestration
│   ├── crawl_result.py     # CrawlResult dataclass
│   ├── output_formats.py   # OutputFormatter (JSON, CSV, etc.)
│   ├── typings.py          # Type aliases (SelectorType, etc.)
│   └── news/
│       ├── real/           # 9 site-specific crawler classes (VnExpress, DanTri, etc.)
│       └── fake/           # Empty — no fake-source crawlers yet
├── models/
│   ├── base.py             # BaseModel, MultimodalModel, FastCNN, ContrastiveLoss
│   ├── coolant.py          # COOLANT (research implementation)
│   ├── coolant_official.py # COOLANT_Official (paper-faithful)
│   ├── resnet_coolant.py   # PatchedCOOLANT / ResNetCOOLANT adapter
│   ├── clip_model.py       # CLIP model
│   ├── senet.py            # SEAttentionModule
│   ├── config.py           # Typed dataclass configs
│   ├── factory.py          # ModelFactory + ModelBuilder
│   └── README.md           # Model documentation
├── preprocessing/
│   ├── text_preprocessing.py    # TextPreprocessor (PhoBERT/ViSoBERT)
│   ├── image_preprocessing.py   # ImagePreprocessor (ResNet/CLIP/SigLIP)
│   ├── combined_preprocessing.py
│   ├── data_utils.py
│   └── example_preprocessing.py
├── processing/
│   ├── vifactcheck_processor.py # ViFactCheck end-to-end dataset processor
│   ├── pytorch_dataset.py       # FakeNewsDataset(Dataset)
│   ├── hdf5_dataset.py          # HDF5-backed dataset
│   ├── multimodal_processor.py  # General multimodal processing
│   ├── image_processor.py
│   ├── text_processor.py
│   ├── simple_dataloader.py
│   └── coolant/                 # COOLANT-specific dataset/pairs
│       ├── pair_dataset.py
│       ├── pair_extractor.py
│       └── training_utils.py
├── helpers/
│   ├── httpx_client.py          # BaseClient (async httpx with retry + legacy SSL)
│   ├── string_handle.py         # StringHandler utility class
│   ├── json_helper.py
│   ├── logger.py                # Centralized logger
│   ├── legacy_tool_handler.py
│   └── file_handler/            # File I/O utilities
├── parser/
│   ├── base.py                  # Base parser
│   └── html_tag_parser.py       # HTML tag extraction
├── exceptions/
│   └── string_exception.py      # URLFormatException, InvalidExtensionException
├── utils/
│   └── device.py                # get_device() — CUDA/MPS/CPU auto-detect
└── lib/
    └── Mocheg/                  # Embedded git submodule (separate codebase)
```

## `tests/` Layout

```
tests/
├── conftest.py             # Empty placeholder
├── crawler/
│   └── test_simple_crawler.py   # Empty placeholder
└── helpers/
    ├── test_string_handle.py    # StringHandler unit tests (parametrized)
    ├── test_json_helper.py
    └── test_data.json
```

## `notebooks/` Layout

```
notebooks/
├── 1_crawl_only.ipynb
├── 2_preprocess_only.ipynb
├── 3_load_dataset_and_train.ipynb
├── 4_train_model.ipynb          # Main training notebook
├── dataset/
│   ├── crawl_data.ipynb
│   └── crawl_vifactcheck.py
└── mlruns/                      # MLflow run artifacts
```

## Key File Locations

| What | Where |
|---|---|
| Entry point | `src/main.py` |
| Primary model | `src/models/coolant_official.py` |
| Model configs | `src/models/config.py` |
| Crawler factory | `src/crawler/crawler_factory.py` |
| HTTP client | `src/helpers/httpx_client.py` |
| Device utils | `src/utils/device.py` |
| ViFactCheck processor | `src/processing/vifactcheck_processor.py` |
| Training notebook | `notebooks/4_train_model.ipynb` |
| SSL config | `openssl.cnf` (root) |
