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

-   Python 3.13+
-   Conda (recommended) or pip

### Setup with Conda (Recommended)

1. Clone the repository:

    ```bash
    git clone https://github.com/ltruonghai257/fake-new-detection.git
    cd fake-new-detection
    ```

2. Create and activate the conda environment:

    ```bash
    conda env create -f environment.yml
    conda activate fake_news
    ```

### Alternative Setup with pip

1. Create a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

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
