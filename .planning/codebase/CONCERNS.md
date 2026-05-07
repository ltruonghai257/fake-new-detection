# CONCERNS.md — Technical Debt & Issues
_Last mapped: 2026-05-08_

## Security

- **SSL verification disabled globally** — `ssl_ctx.check_hostname = False; ssl_ctx.verify_mode = ssl.CERT_NONE` in `src/helpers/httpx_client.py:29-30`. Intentional for legacy Vietnamese sites, but applies to ALL requests including HuggingFace downloads. MITM risk.
- **`verify=True` parameter ignored** — `BaseCrawler.__init__` passes `verify=True` to `BaseClient`, but `BaseClient.__init__` ignores the parameter and always creates the permissive SSL context.
- **OPENSSL_CONF global mutation** — `src/main.py` sets `os.environ["OPENSSL_CONF"]` before any imports. This affects the entire process environment.

## Architecture / Design

- **Dual COOLANT implementations** (`coolant.py` vs `coolant_official.py`) — unclear which is canonical for training; `factory.py:create_coolant_model()` returns `COOLANT_Official` despite the function name suggesting otherwise.
- **`src/crawler/news/fake/` is empty** — fake news source crawlers not implemented. The ViFactCheck dataset includes fake URLs from unknown domains that will always result in "No crawler found" failures.
- **`src/lib/Mocheg/` is an embedded git submodule** with its own `.git/` — not integrated with the main COOLANT pipeline. Unclear if it's actively maintained or just vendored for reference.
- **`src/crawler/base.py` is 0 bytes** — empty file, likely vestigial.
- **`sys.path.insert(0, ...)` hacks** — `src/preprocessing/text_preprocessing.py` and `image_preprocessing.py` manually insert parent path for `utils.device` import. This is fragile and order-dependent.
- **`SENet` model not implemented** — `ModelFactory._create_senet()` raises `NotImplementedError`. Config exists but no implementation.

## Technical Debt

- **`requirements.txt` is unreadable** — contains null bytes (binary content), not a valid pip requirements file. Dependency management relies solely on the sparse `environment.yml` which has no pinned package versions.
- **No pinned dependency versions** — `environment.yml` specifies only the conda prefix with no packages listed, making environment reproduction unreliable.
- **Duplicate code** — `src/processing/vifactcheck_processor.py` and `src/processing/pytorch_dataset.py` both define text/image processing logic that partially duplicates `src/preprocessing/`.
- **Multiple checkpoint directories** — `checkpoints/`, `training/checkpoints/`, `training/checkpoints_coolant/`, `training/checkpoints_coolant_phased/`, `training/checkpoints_phased/` with no clear organization.
- **Test coverage is very low** — only `StringHandler` has meaningful pytest coverage. All model, preprocessing, and crawler code is untested.

## Fragile Areas

- **Crawler state management** — `crawling_status.json` cache stored in the working directory. Multiple concurrent runs or different `cwd` would corrupt the cache.
- **Image hash naming** — images named by content hash. If a site changes an image URL but keeps the same content, the existing file is silently reused. Collision risk is low but not zero.
- **`asyncio.gather` without error isolation** — uncaught exceptions in one `process_url` coroutine could theoretically propagate (though `try/except` in `process_url` mitigates this).
- **HuggingFace model loading at dataset init time** — `ViFactCheckTextProcessor.__init__` downloads/loads PhoBERT during `__init__`. Failure silently sets `self.tokenizer = None` and returns zero tensors, masking the error.

## Performance

- **No text feature caching by default** — `FakeNewsDataset` has `cache_text=False` by default; text re-encoded on every `__getitem__`. For large datasets this is very slow.
- **HDF5 preprocessed files** not always used — `hdf5_dataset.py` exists but several notebooks bypass it and process on-the-fly.
- **`max_concurrent=15`** hardcoded in `crawl_and_save_all()` signature default — reasonable but not tunable via config.

## Known Issues

- **`verify=True` constructor parameter silently ignored** in `BaseClient` — documented behavior gap.
- **`src/models/factory.py:create_coolant_model()`** returns `COOLANT_Official` instead of `COOLANT` — likely intentional but misleading.
- **`.claude/worktrees/`** directory contains multiple stale git worktrees (`compassionate-benz`, `hopeful-noether`, `keen-grothendieck`, `nice-golick`) that are not gitignored and clutter the repo.
