# TESTING.md — Test Structure & Practices
_Last mapped: 2026-05-08_

## Framework

- **pytest** — configured via `pytest.ini` (root and `src/pytest.ini`)
- `tests/conftest.py` — present but empty (no shared fixtures defined)

## Test Layout

```
tests/
├── conftest.py                      # Empty
├── crawler/
│   └── test_simple_crawler.py       # Empty placeholder
└── helpers/
    ├── test_string_handle.py        # Active: StringHandler unit tests
    ├── test_json_helper.py          # Helper tests
    └── test_data.json               # Test fixture data
```

Additional test files at `src/` level (run from `src/` working dir):
- `src/test_crawler.py` — integration test / actual crawler runner (`main()` called by `src/main.py`)
- `src/agent_test.py` — integration tests for specific crawler logic
- `src/test_scripts.py` — basic utility function tests
- `src/4_train_model.ipynb` — notebook-based training test

Root-level test files:
- `test_existing_splits.py`
- `test_multimodal_vietnamese.py`
- `test_src_preprocessing.py`
- `test_deep_crawling.ipynb`

## What Is Tested

| Area | Status | Notes |
|---|---|---|
| `StringHandler` | Active, parametrized | `is_url`, `sanitize_filename`, `class_name_to_snake_case`, `count_words` |
| `JsonHelper` | Present | `test_json_helper.py` |
| Crawlers | Placeholder only | `test_simple_crawler.py` is empty |
| Models | No unit tests | No pytest files for COOLANT, CLIP, etc. |
| Preprocessing | Root-level scripts | `test_src_preprocessing.py`, `test_multimodal_vietnamese.py` |
| Dataset splits | Root-level script | `test_existing_splits.py` |

## Test Style

Parametrized tests with `@pytest.mark.parametrize`:

```python
@pytest.mark.parametrize("input_string, expected", [
    ("http://example.com", True),
    ("example.com", False),
    ...
])
def test_is_url(self, input_string, expected):
    assert StringHandler.is_url(input_string) == expected
```

- Tests organized as classes (`TestStringHandle`)
- Import path: `from src.helpers import StringHandler` (uses `src.` prefix)

## Running Tests

```bash
# From project root
python -m pytest

# From src/ (for src-relative imports)
cd src && python -m pytest
```

## Gaps / Coverage

- **No model unit tests** — COOLANT, CLIP, SENet have zero pytest coverage
- **No preprocessing unit tests** — `TextPreprocessor`, `ImagePreprocessor` not tested
- **Crawler integration tests** — `test_simple_crawler.py` is empty; real testing done via `agent_test.py` manually
- **No CI pipeline** — no `.github/workflows/` or similar
- **No mocking** — no `unittest.mock` or `pytest-mock` usage found
- **No coverage reporting** — no `pytest-cov` configuration
