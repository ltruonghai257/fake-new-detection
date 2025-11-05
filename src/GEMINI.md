# Fake News Detection Crawler

## Project Overview

This project is a Python-based web crawler designed to collect data for a fake news detection system. It fetches URLs from a Hugging Face dataset, crawls the corresponding news articles, and saves the content for further analysis.

The project is built using the following technologies:

- **Python 3.13**
- **Asyncio** for asynchronous programming
- **httpx** for making HTTP requests
- **BeautifulSoup** and **lxml** for parsing HTML
- **datasets** for loading data from Hugging Face
- **tqdm** for progress bars

## Building and Running

### Prerequisites

- Python 3.13
- Pip

### Installation

1.  Clone the repository.
2.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Running the Crawler

To run the crawler, execute the following command:

```bash
python main.py
```

This will start the crawling process, fetching URLs from the `tranthaihoa/vifactcheck` dataset on Hugging Face. The crawled data will be saved in the `data/json` directory.

### Running Tests

To run the tests, execute the following command:

```bash
python test_crawler.py
```

## Development Conventions

- **Coding Style:** The project follows the PEP 8 style guide for Python code.
- **Testing:** The project uses the `unittest` framework for testing. All tests are located in the `tests` directory.
- **Contributions:** Contributions are welcome. Please open an issue or submit a pull request.
