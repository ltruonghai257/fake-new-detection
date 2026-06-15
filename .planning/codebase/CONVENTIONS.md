# CONVENTIONS.md — Code Conventions & Patterns

## Language Style
- **Python 3.10+** — uses `match/case`, structural `TypedDict`, `|` union types
- **Type hints** used throughout: `List`, `Dict`, `Optional`, `Tuple`, `TypedDict`, `Callable`
- **Dataclasses / TypedDicts** for config and structured data (e.g. `TextCleaningOptions`, model configs)
- Docstrings follow **Google style** (some files), plain prose others — inconsistent across modules

## Module Structure
- Public API exposed via `__init__.py` with explicit `__all__` lists (see `src/models/__init__.py`)
- Internal imports use relative paths within packages; cross-package imports use absolute `src`-rooted paths
- `sys.path.insert` workaround used in some files for cross-package imports (e.g. `text_preprocessing.py` line 21)

## Design Patterns
- **Factory Pattern** — `CrawlerFactory` (crawler routing), `ModelFactory` / `ModelBuilder` (model creation)
- **Strategy / Template Method** — `BaseCrawler` defines abstract selector properties; subclasses implement
- **Builder** — `ModelBuilder.build()` chains configuration into model instantiation
- **Optional dependency guards** — `try/except ImportError` pattern for `underthesea`, `py_vncorenlp`

## Naming Conventions
- Classes: `PascalCase` (e.g. `VnExpressCrawler`, `CrawlJournal`, `EncodingPart`)
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE` (e.g. `CRAWLER_MAPPING`, `AVAILABLE_MODELS`)
- Config dataclasses: `*Config` suffix (e.g. `COOLANTConfig`, `TrainingConfig`)
- Test files: `test_*.py` prefix

## Async Patterns
- Crawler is fully async (`asyncio` + `httpx.AsyncClient`)
- Concurrency limited via `asyncio.Semaphore`
- `nest-asyncio` installed for Jupyter cell compatibility
- `tqdm.asyncio` used for async progress bars

## Logging
- **Loguru** (`from helpers.logger import logger`) — single shared logger singleton
- Log levels: `logger.info(...)`, `logger.warning(...)`, `logger.error(...)`
- Log output: `logs/app.log` (configured in `helpers/logger.py`)

## Configuration / Secrets
- All paths/secrets via **environment variables** loaded with `python-dotenv`
- `DATA_ROOT` env var drives all data path resolution (`helpers/paths.py → get_data_root()`)
- Multiple `.env.*` files — never committed; `.env.*.example` files committed as templates

## Error Handling
- Network errors: caught at `BaseCrawler.fetch()` level, logged to `CrawlJournal.failed`
- SSL errors: handled globally via `OPENSSL_CONF` env var (not per-request)
- Missing crawlers: logged as `"No crawler found"` in failed URLs JSON

## Model Code Style
- All models inherit `nn.Module`
- `forward()` methods typed with input/output tensor annotations
- Config passed as dataclass (not raw dicts)
- `ModelFactory.create_model(name, config)` as the canonical instantiation path
