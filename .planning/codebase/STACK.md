# STACK.md — Technology Stack

## Language & Runtime
- **Python 3.10+** (pyproject.toml `requires-python = ">=3.10"`)
- **uv** for dependency management & virtual environments (`uv sync`)
- **Jupyter / JupyterLab** for interactive notebooks (`notebooks/pipeline/`)

## Deep Learning
| Library | Version | Role |
|---------|---------|------|
| PyTorch | `>=2.12.0` | Core tensor ops, model training |
| torchvision | `>=0.27.0` | ResNet image encoders, transforms |
| transformers | `>=4.40.0` | PhoBERT / BERT tokenization & embeddings |
| tokenizers | `>=0.19.0` | HuggingFace fast tokenizers |
| datasets | `>=2.19.0` | HuggingFace dataset loading |
| huggingface-hub | `>=0.23.0` | Model/dataset hub access |
| safetensors | `>=0.4.3` | Safe checkpoint serialization |

## Vietnamese NLP
- **underthesea** `>=6.8.0` — Vietnamese word tokenization, text normalization, accent fixing
- **PhoBERT** (`vinai/phobert-base`) — Vietnamese BERT pre-trained model (via transformers)
- Optional: **VnCoreNLP** (`py_vncorenlp`) — alternative Vietnamese NLP toolkit

## Data / ML Utilities
- **numpy** `<2`, **pandas** `>=2.0.0`, **scikit-learn** `>=1.3.0`
- **pyarrow** `>=15.0.0`, **h5py** `>=3.11.0` — columnar/HDF5 data formats
- **Pillow** `>=10.3.0` — image loading & conversion to JPG

## Web Crawling & HTTP
- **httpx** `>=0.27.0` — async HTTP client (primary crawler transport)
- **requests** `>=2.31.0` — sync fallback
- **beautifulsoup4** + **lxml** — HTML DOM parsing & CSS selector extraction
- **orjson** `>=3.10.0` — fast JSON serialization
- Custom `openssl.cnf` injection to handle `UNSAFE_LEGACY_RENEGOTIATION_DISABLED` on legacy Vietnamese gov sites

## Experiment Tracking & Visualization
- **MLflow** `>=2.11.0` — experiment tracking (runs stored in `archive/mlruns/`)
- **matplotlib**, **seaborn** — plotting
- **papermill** `>=2.7.0` — parameterized notebook execution

## External Integrations
- **Google Drive API** (`google-api-python-client`, `google-auth-*`) — remote dataset upload
- **Vast.ai** (`vastai/`) — remote GPU training scripts

## Dev & Test
- **pytest** `>=8.0` — test runner (`pyproject.toml` config)
- **loguru** `>=0.7.2` — structured logging
- **tqdm** `>=4.66.0` — progress bars
- **python-dotenv** `>=1.0.0` — env management (`.env.*` variants per environment)
- **nest-asyncio** `>=1.6.0` — async in Jupyter cells
