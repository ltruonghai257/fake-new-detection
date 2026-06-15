# INTEGRATIONS.md — External Integrations & Data Sources

## Dataset: ViFactCheck
- **What:** Vietnamese fact-checking benchmark dataset (URLs + labels)
- **Role:** Primary source of URLs fed into the crawler
- **Label file:** `news_data.json` (root), split into train/test/dev
- **Reference:** `docs/papers/2412.15308v1.pdf`

## HuggingFace Hub
- **PhoBERT** (`vinai/phobert-base`) — downloaded via `transformers.AutoModel`
- Checkpoints cached locally; also uploadable to HuggingFace Hub via `huggingface-hub`

## Vietnamese News Sites (Crawlers)
Site-specific crawlers in `src/crawler/news/real/`:
| Domain | Crawler Class |
|--------|--------------|
| vnexpress.net | `VnExpressCrawler` |
| baochinhphu.vn | `BaoChinhPhuCrawler` |
| dantri.com.vn | `DanTriCrawler` |
| nguoilaodong.com.vn | `NguoiLaoDongCrawler` |
| tuoitre.vn | `TuoiTreCrawler` |
| baotintuc.vn | `BaoTinTucCrawler` |
| phapluattp.vn | `PhapLuatHcmCrawler` |
| thanhnien.vn | `ThanhNienCrawler` |
| tienphong.vn | `TienPhongCrawler` |

One fake-news source directory (`src/crawler/news/fake/`) also exists.

## Google Drive
- **Library:** `google-api-python-client` + OAuth2 (`google-auth-oauthlib`)
- **Usage:** Upload processed datasets and checkpoints to shared Drive
- **Entry point:** `notebooks/google_drive_upload.ipynb`, `src/helpers/google_drive_uploader.py`
- **Credentials:** loaded from `.env` / service account JSON (not committed)

## Vast.ai (Remote GPU)
- **Directory:** `vastai/`
- **Scripts:** `autosync_vastai.py`, `download_from_vastai.py`, `setup_vastai.sh`
- **Guide:** `vastai/VASTAI_GUIDE.md`
- **Pattern:** local → rsync to Vast.ai instance → train → rsync back

## MLflow
- **Tracking:** local filesystem (`archive/mlruns/`)
- **Usage:** Experiment run logging in training notebooks
- **Note:** Pinned to `protobuf==3.20.3` due to descriptor conflict with older MLflow

## OpenSSL Override
- **File:** `src/openssl.cnf`
- **Why:** Many Vietnamese government/local news sites use legacy TLS renegotiation. `src/main.py` sets `OPENSSL_CONF` env var to point at this file before crawling starts.

## Environment Configurations
Multiple `.env.*` files for different deployment targets:
- `.env.mac` — local macOS development
- `.env.vastai.example` — Vast.ai GPU server
- `.env.colab.example` — Google Colab
- `.env.windows.example` — Windows dev machine
