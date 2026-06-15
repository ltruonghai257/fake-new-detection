# Fake News Detection Crawler

## Project Overview

This project is a sophisticated Python-based web crawler designed to collect and process data for a fake news detection system. It features a modular architecture for crawling Vietnamese news sites, with built-in support for image extraction and flexible content parsing.

### Key Features

-   Asynchronous crawling with robust error handling
-   Modular crawler architecture supporting multiple news sites
-   Built-in image extraction and storage
-   Flexible HTML parsing with CSS and XPath selectors
-   JSON-based data storage with configurable output formats
-   Comprehensive test suite with pytest

## Technology Stack

### Core Dependencies

-   **Python 3.13.9**
-   **httpx 0.28.1** - Async HTTP client with modern features
-   **BeautifulSoup4 4.14.2** and **lxml 5.4.0** - HTML parsing
-   **aiofiles 25.1.0** - Async file operations
-   **Pillow 12.0.0** - Image processing
-   **tqdm 4.67.1** - Progress tracking

### Development Tools

-   **pytest** - Testing framework
-   **playwright 1.55.0** - Web automation (optional)
-   **fake-useragent 2.2.0** - User agent rotation

## Project Structure

```plaintext
src/
├── crawler/          # Core crawler implementations
│   ├── base_crawler.py     # Base crawler class
│   ├── base.py            # Crawler registry
│   └── news/             # News-specific crawlers
├── helpers/         # Utility functions
│   ├── httpx_client.py    # Custom HTTP client
│   └── file_handler/     # File handling modules
└── parser/          # HTML parsing modules
```

## Installation

### Prerequisites

-   Python 3.10+
-   [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Setup with uv

1. Clone the repository:

    ```bash
    git clone https://github.com/ltruonghai257/fake-new-detection.git
    cd fake-new-detection
    ```

2. Install dependencies and activate the environment:

    ```bash
    uv sync
    source .venv/bin/activate
    ```

> **Note**: `torch`/`torchvision` are not included in `uv sync` — install them separately
> with the appropriate CUDA index URL (see `vastai/setup_vastai.sh` for reference).

## Usage

### Running the Crawler

1. Basic crawling:

    ```bash
    python src/main.py
    ```

2. Jupyter notebooks for testing:

    ```bash
    jupyter notebook notebooks/test_crawler.ipynb
    ```

### Configuration

The crawler supports various configuration options in `config.json`:

-   Output formats (JSON/CSV)
-   Image storage settings
-   Crawling depth and limits
-   Site-specific settings

### Running Tests

Run the test suite:

```bash
python -m pytest
```

## Development Guidelines

### Adding a New Crawler

1. Create a new crawler class in `src/crawler/news/real/`:

    ```python
    from src.crawler.base_crawler import BaseCrawler

    class MySiteCrawler(BaseCrawler):
        title_selector = {"css_selector": ["h1.title"]}
        content_selector = {"css_selector": [".article-content"]}
        # ... other selectors
    ```

2. Register in `src/crawler/base.py`:

    ```python
    CRAWLER_MAPPING = {
        "mysite.com": MySiteCrawler
    }
    ```

### Code Style

-   Follow PEP 8 guidelines
-   Use type hints
-   Document public APIs
-   Write test cases for new features

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License

This project is open source and available under the MIT License.

<!-- CODEGRAPH_START -->
## CodeGraph

In repositories indexed by CodeGraph (a `.codegraph/` directory exists at the repo root), reach for it BEFORE grep/find or reading files when you need to understand or locate code:

- **MCP tools** (when available): `codegraph_explore` answers most code questions in one call — the relevant symbols' verbatim source plus the call paths between them. `codegraph_node` returns one symbol's source + callers, or reads a whole file with line numbers. If the tools are listed but deferred, load them by name via tool search.
- **Shell** (always works): `codegraph explore "<symbol names or question>"` and `codegraph node <symbol-or-file>` print the same output.

If there is no `.codegraph/` directory, skip CodeGraph entirely — indexing is the user's decision.
<!-- CODEGRAPH_END -->
