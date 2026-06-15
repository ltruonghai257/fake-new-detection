# TESTING.md — Testing Strategy & Coverage

## Test Framework
- **pytest** `>=8.0` — configured in `pyproject.toml`
- Config: `addopts = "-ra -v --disable-warnings"`, `testpaths = ["tests"]`
- `pythonpath = [".", "src"]` — both project root and `src/` on path

## Test Structure

```
tests/
├── conftest.py                     # Empty — shared fixtures TBD
├── crawler/
│   └── test_simple_crawler.py     # Empty stub — no crawler tests yet
├── helpers/
│   ├── test_data.json             # Fixture data
│   ├── test_json_helper.py        # Unit tests for extract_fields_from_json
│   └── test_string_handle.py      # Unit tests for string helpers
└── processing/
    └── coolant/
        └── test_pair_extractor.py # Unit tests for PairExtractor
```

## What Is Tested

### `tests/helpers/test_json_helper.py`
- `extract_fields_from_json()` — field extraction, missing fields, empty fields, nonexistent fields
- Fixture: inline `test_data.json` with 3 articles

### `tests/helpers/test_string_handle.py`
- String manipulation helpers from `src/helpers/string_handle.py`

### `tests/processing/coolant/test_pair_extractor.py`
- `PairExtractor.extract_from_json()` — backward compatibility (returns list)
- `return_stats=True` mode — returns `(pairs, stats)` tuple
- `pair_text` field construction (title + caption concatenation)
- `source_label_counts` tracking (real/fake split stats)
- Caption filtering: credit-only captions (e.g. "Ảnh: NVCC") are skipped (`no_caption` stat)
- Uses `tempfile.TemporaryDirectory` + fake image files for isolation

## What Is NOT Tested (Coverage Gaps)
- **Crawler logic** — `test_simple_crawler.py` is empty; no tests for `BaseCrawler`, `CrawlerFactory`, `CrawlJournal`, or any site-specific crawler
- **Model code** — zero tests for `COOLANT`, `CLIP`, `SENet`, `ModelFactory`
- **Preprocessing** — no tests for `text_preprocessing.py`, `image_preprocessing.py`, `combined_preprocessing.py`
- **conftest.py** — empty; no shared fixtures defined

## Test Markers
- `slow` — marks slow-running tests (`@pytest.mark.slow`)
- `integration` — marks integration tests (`@pytest.mark.integration`)
Neither marker is currently used in any test file.

## Running Tests
```bash
# All tests
pytest

# With verbose output
pytest -v

# Specific file
pytest tests/processing/coolant/test_pair_extractor.py

# Skip slow tests
pytest -m "not slow"
```

## Notes
- `test_pair_extractor.py` uses `importlib.util` to load `pair_extractor.py` from an absolute path — this works but is fragile; paths assume `src/processing/coolant/` exists (note: actual location is `src/preprocessing/coolant/`)
- No mocking framework (e.g. `unittest.mock`) currently in use
