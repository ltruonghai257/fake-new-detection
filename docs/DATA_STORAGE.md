# Data Storage Guide

Two complementary features for managing large files that are excluded from git:

1. **External Data Root (`DATA_ROOT`)** — redirect all pipeline output to a local drive
2. **Google Drive Sync** — upload/download those files across machines

---

## 1. External Data Root (`DATA_ROOT`)

### Why

The following directories are git-ignored because they contain large binary files:

| Directory | Contents |
|---|---|
| `data/json/` | Crawled news article JSON per split |
| `data/jpg/` | Downloaded article images |
| `processed_data/hdf5/` | COOLANT HDF5 datasets |
| `training/checkpoints_coolant/` | Stage 1 model checkpoints |
| `training/checkpoints_stage2/` | Stage 2 model checkpoints |
| `training/stage2_features/` | Extracted feature tensors |
| `training/stage2_results/` | Evaluation outputs |
| `mlruns/` | MLflow experiment tracking |

By default these all live inside the repo root. Setting `DATA_ROOT` moves them to any path — e.g. an external SSD — without touching any source code.

### How it works

Every pipeline notebook and `pipeline_config.py` resolve `DATA_ROOT` with the same logic:

```python
DATA_ROOT = Path(os.environ["DATA_ROOT"]) if os.environ.get("DATA_ROOT") else PROJECT_ROOT
```

Priority: **`DATA_ROOT` env var** → **repo root** (backward-compatible default).

### Setup

**Step 1 — Copy `.env.example` to `.env` and set `DATA_ROOT`:**

```bash
cp .env.example .env
# then edit .env:
DATA_ROOT=/Volumes/MyDrive/fake-news-data
```

**Step 2 — Load `.env` before launching Jupyter:**

```bash
export $(grep -v '^#' .env | grep '.' | xargs) && jupyter lab
```

Or add the export to your shell profile / conda activation script so it applies automatically.

**Step 3 — Verify** (run in any notebook or terminal):

```python
import os
from pathlib import Path
print(Path(os.environ.get("DATA_ROOT", ".")))
```

### Practical setups by environment

#### macOS with Google Drive mounted (current local workflow)

Google Drive client mounts your Drive as a local path. Set `DATA_ROOT` directly to that path — no upload/download step needed; the OS handles syncing transparently.

```bash
# .env on macOS
DATA_ROOT=/Users/<you>/Library/CloudStorage/GoogleDrive-<email>/My Drive/Thesis_Final/fake-news-data-for-thesis
```

All pipeline output (JSON, images, caches, checkpoints) writes straight to Drive.

#### Vast.ai (training only)

Google Drive is not mounted on a remote GPU instance. Use **rclone** to pull preprocessed data before training and push checkpoints after. See [`vastai/VASTAI_GUIDE.md`](../vastai/VASTAI_GUIDE.md#data-from-google-drive-rclone) for the full rclone walkthrough.

```bash
# .env on Vast.ai (copy from .env.vastai.example)
DATA_ROOT=/workspace/fake-news-data-for-thesis
```

### Migrate existing data to the new root

If you already have data inside the repo and want to move it out, run once:

```bash
python migrate_data_root.py /Volumes/MyDrive/fake-news-data
```

What `migrate_data_root.py` moves:

| Source (repo-relative) | Destination |
|---|---|
| `data/` | `<new_root>/data/` |
| `processed_data/` | `<new_root>/processed_data/` |
| `training/` | `<new_root>/training/` |
| `notebooks/data/` | `<new_root>/data/` (merged) |
| `notebooks/mlruns/` | `<new_root>/mlruns/` |

Existing destinations are never overwritten — only missing items are moved.

### Where `DATA_ROOT` is consumed

| File | Variable |
|---|---|
| `notebooks/pipeline/pipeline_config.py` | `make_config(project_root, data_root=None)` |
| `notebooks/pipeline/01_data_crawling.ipynb` | `DATA_ROOT` in config cell |
| `notebooks/pipeline/02_preprocessing.ipynb` | `DATA_ROOT` in config cell |
| `notebooks/pipeline/03_coolant_training.ipynb` | `DATA_ROOT` in config cell |
| `config.json` | `output.data_root` key (null = use repo root) |

---

## 2. Google Drive Sync

### Why

When switching machines, `git pull` restores source code but not the large binary files above. Google Drive sync replaces manual USB transfers or rsync over SSH.

### Setup

**Option A — Service account (recommended for automation)**

1. Go to [Google Cloud Console](https://console.cloud.google.com/) → IAM → Service Accounts → Create.
2. Download the JSON key file. Place it outside the repo (it is git-ignored via `.env`).
3. Share the target Drive folder with the service account email address.
4. Set `GOOGLE_APPLICATION_CREDENTIALS` in `.env`:
   ```
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
   ```

**Option B — OAuth (interactive, personal use)**

1. Google Cloud Console → APIs & Services → Credentials → Create OAuth client ID (Desktop app).
2. Download the JSON file.
3. Set `CREDENTIALS_PATH` in the notebook config cell to point at it.
4. Set `USE_OAUTH = True`.  A browser window opens on the first run; a `token.pickle` is cached next to the credentials file.

### API reference

Module: `src/helpers/google_drive_uploader.GoogleDriveUploader`

```python
from helpers.google_drive_uploader import GoogleDriveUploader

uploader = GoogleDriveUploader(
    credentials_path="service-account-key.json",  # or set GOOGLE_APPLICATION_CREDENTIALS
    use_oauth=False,
)
```

#### `upload_file(local_path, folder_path=None, overwrite=False) → str`

Upload a single file. Returns the Drive file ID.

```python
file_id = uploader.upload_file(
    "data/json/news_data_vifactcheck_train.json",
    folder_path="fake-news-detection/data",
    overwrite=True,
)
```

#### `upload_directory(local_dir, folder_path=None, extensions=None, recursive=False, overwrite=False) → list[str]`

Upload all matching files from a directory. Returns a list of Drive file IDs.

```python
ids = uploader.upload_directory(
    local_dir="training/checkpoints_coolant",
    folder_path="fake-news-detection/models",
    extensions=[".pth", ".h5", ".safetensors"],
    recursive=True,
    overwrite=True,
)
```

#### `download_file(file_id, local_path) → None`

Download a file by its Drive file ID.

```python
uploader.download_file("1abc...xyz", "training/checkpoints_coolant/best_model.pth")
```

#### `download_folder(folder_path, local_dir, extensions=None, recursive=False) → list[str]`

Download all matching files from a Drive folder, mirroring folder structure.

```python
paths = uploader.download_folder(
    folder_path="fake-news-detection/data",
    local_dir="data/json",
    extensions=[".json"],
    recursive=True,
)
```

### Supported file types

| Category | Extensions |
|---|---|
| JSON data | `.json` |
| Images | `.jpg` `.jpeg` `.png` `.gif` `.bmp` `.webp` `.svg` |
| Text / docs | `.txt` `.md` `.csv` |
| Model files | `.h5` `.hdf5` `.pkl` `.pt` `.pth` `.safetensors` `.npz` `.ckpt` `.bin` |
| Archives | `.zip` |

Any unrecognised extension falls back to `application/octet-stream`.

### Notebook

`notebooks/google_drive_upload.ipynb` provides ready-to-run cells for the full cross-machine workflow:

| Section | Purpose |
|---|---|
| **Config** | Set `CREDENTIALS_PATH`, `USE_OAUTH`, `DRIVE_ROOT_FOLDER` |
| **U1 — Upload JSON data** | `data/json/` → Drive |
| **U2 — Upload images** | `data/jpg/` → Drive |
| **U3 — Upload model files** | `training/checkpoints*/` → Drive |
| **D1 — Download JSON data** | Drive → `data/json/` |
| **D2 — Download images** | Drive → `data/jpg/` |
| **D3 — Download model files** | Drive → `training/checkpoints*/` |

---

## Cross-machine workflow (combined)

```
Machine A                              Machine B
──────────────────────────────         ──────────────────────────────
1. git push                      →     1. git pull
2. Run UPLOAD cells in           →     2. export DATA_ROOT=...
   google_drive_upload.ipynb            3. Run DOWNLOAD cells in
                                           google_drive_upload.ipynb
```

Both machines must have a copy of the credentials file (copy manually — never commit it).
