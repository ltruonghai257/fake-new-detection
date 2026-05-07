# CONVENTIONS.md — Code Style & Patterns
_Last mapped: 2026-05-08_

## Language Style

- **Python 3.8+** with type hints throughout (`typing` module, `TypedDict`, `Optional`, `Union`, `Tuple`, `Dict`, `List`)
- `from __future__ import annotations` not used — explicit `Optional[X]` style preferred
- Dataclasses (`@dataclass`) used for config objects (`src/models/config.py`)
- Abstract base classes (`ABC`, `@abstractmethod`) used for `BaseCrawler` and `MultimodalModel`

## Naming Conventions

| Entity | Convention | Example |
|---|---|---|
| Classes | `PascalCase` | `CrawlerFactory`, `VnExpressCrawler`, `COOLANTConfig` |
| Functions/methods | `snake_case` | `get_crawler()`, `crawl_and_save_all()` |
| Constants | `UPPER_SNAKE_CASE` | `CRAWLER_MAPPING`, `TEXT_MODEL_REGISTRY` |
| Private helpers | `_snake_case` | `_save_checkpoint()`, `_init_text_models()` |
| File names | `snake_case.py` or `PascalCaseCrawler.py` for crawlers | `crawler_factory.py`, `VnExpressCrawler.py` |

## Factory Pattern (dominant pattern)

All extensible component types use factory routing:

```python
# Crawler factory — domain string → class
CRAWLER_MAPPING = {
    "vnexpress.net": VnExpressCrawler,
    ...
}

# Model factory — name string → class
_model_registry = {
    "coolant": COOLANT,
    "clip": CLIP,
    ...
}
```

Adding a new crawler = create file + add one entry to `CRAWLER_MAPPING`. No core changes.

## Abstract Contract Pattern

Crawlers must implement four abstract properties:

```python
class BaseCrawler(ABC):
    @property @abstractmethod
    def title_selector(self) -> SelectorType: ...
    @property @abstractmethod
    def image_selector(self) -> SelectorType: ...
    @property @abstractmethod
    def content_selector(self) -> SelectorType: ...
    @property @abstractmethod
    def link_selector(self) -> SelectorType: ...
```

Models must implement three abstract methods:

```python
class MultimodalModel(BaseModel):
    @abstractmethod def encode_text(...): ...
    @abstractmethod def encode_image(...): ...
    @abstractmethod def fuse_modalities(...): ...
```

## Error Handling

- Custom exception classes in `src/exceptions/` (`URLFormatException`, `InvalidExtensionException`)
- HTTP errors: `try/except httpx.HTTPStatusError` with retry loop in `BaseClient._request()`
- Missing crawlers: logged as `"No crawler found"` in `failed_urls.json`, never raised
- Optional imports wrapped in `try/except ImportError` with availability flag (`UNDERTHESEA_AVAILABLE`, `CV2_AVAILABLE`)
- Model loading: graceful fallback `logger.warning()` when HuggingFace model unavailable

## Logging

- Centralized logger: `from helpers.logger import logger` (uses `logging` module)
- No `print()` in production code — all output via `logger.info/warning/error`
- `src/utils/device.py` uses module-level `logger = logging.getLogger(__name__)`
- Log files: `logs/app.log`

## Async Style

- `asyncio` + `httpx.AsyncClient` for all network I/O
- `asyncio.Semaphore` for concurrency control
- `asyncio.Lock` for shared state mutation (result lists, counters)
- Async methods prefixed `a` by convention: `arun()`, `_save_images()`
- `asyncio.gather(*tasks)` for parallel crawling

## Config Pattern

Typed dataclasses with `to_dict()` / `from_dict()` + factory functions:

```python
config = get_coolant_config(learning_rate=1e-3, batch_size=64)
# or
config = COOLANTConfig(shared_dim=128, sim_dim=64)
model = COOLANT(config.to_dict())
```

## Module-Level Registry

Models and preprocessors use module-level registry dicts:

```python
TEXT_MODEL_REGISTRY = {
    "phobert-base": {"hf_id": "vinai/phobert-base", "feature_dim": 768},
    ...
}
IMAGE_MODEL_REGISTRY = {
    "resnet50": {"family": "resnet", "feature_dim": 2048, ...},
    ...
}
```

## Import Style

- Relative imports within packages: `from .base_crawler import BaseCrawler`
- Absolute imports from `src/` root when cross-package: `from helpers.logger import logger`
- `sys.path.insert(0, ...)` used in preprocessing files to resolve `src/` root (fragile — see CONCERNS)
