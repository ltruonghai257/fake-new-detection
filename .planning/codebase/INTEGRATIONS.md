# INTEGRATIONS.md — External Services & APIs
_Last mapped: 2026-05-08_

## Vietnamese News Sources (Crawler Targets)

These are the external HTTP endpoints the crawler hits. All go through `src/helpers/httpx_client.py`.

| Domain | Crawler Class | Notes |
|---|---|---|
| `vnexpress.net` | `VnExpressCrawler` | Major general news |
| `baochinhphu.vn` | `BaoChinhPhuCrawler` | Government site — needs legacy SSL |
| `dantri.com.vn` | `DanTriCrawler` | Major general news |
| `nld.com.vn` | `NguoiLaoDongCrawler` | Labour newspaper |
| `tuoitre.vn` | `TuoiTreCrawler` | Major general news |
| `baotintuc.vn` | `BaoTinTucCrawler` | Tin Tức newspaper |
| `plo.vn` | `PhapLuatHcmCrawler` | Legal/HCM news |
| `thanhnien.vn` | `ThanhNienCrawler` | Youth newspaper |
| `tienphong.vn` | `TienPhongCrawler` | Tiền Phong newspaper |

No fake-news source crawlers implemented yet (`src/crawler/news/fake/` is empty).

## HuggingFace Hub

- **Downloaded at runtime** via `transformers.AutoTokenizer.from_pretrained()` and `AutoModel.from_pretrained()`
- Models: `vinai/phobert-base`, `vinai/phobert-base-v2`, `vinai/phobert-large`, `uitnlp/visobert`
- Also: `openai/clip-vit-large-patch14`, `google/siglip-base-patch16-224`
- **No API key required** — public HF models

## SSL / Network

- Legacy SSL override via `openssl.cnf` + `os.environ["OPENSSL_CONF"]` set in `src/main.py`
- `ssl.OP_LEGACY_SERVER_CONNECT` flag applied in `BaseClient` (`src/helpers/httpx_client.py:28`)
- `check_hostname = False`, `verify_mode = CERT_NONE` used for resilience against bad certs
- Retry logic: 3 retries, exponential backoff (0.5 × 2^i seconds) on HTTP 5xx

## Experiment Tracking (Mocheg submodule only)

- `mlflow` — used in `src/lib/Mocheg/controllable/` and `src/lib/Mocheg/verification/`
- `wandb` — used in same Mocheg files
- **Not used** in the main COOLANT training pipeline (`src/`)

## File Storage

- **Local filesystem only** — no cloud storage (S3, GCS, etc.)
- JSON crawl outputs: `data/json/news_data_vifactcheck_<split>.json`
- Image outputs: `data/jpg/<site_name>/<hash>.jpg`
- Crawl cache: `crawling_status*.json` (root)
- Failed URLs: `failed_urls*.json` (root)
- Model checkpoints: `checkpoints/`, `training/checkpoints*/`
- HDF5 preprocessed data: via `src/processing/hdf5_dataset.py`

## Dataset

- **ViFactCheck** — primary benchmark dataset; URL lists drive the crawler
  - Split files: `train`, `test`, `dev`
  - Format: JSON with caption, label, image URL fields
