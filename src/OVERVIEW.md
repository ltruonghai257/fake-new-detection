# Vietnamese Multimodal Fake News Detection System

**Version:** 2.0.0
**Status:** Active Development
**Primary Language:** Python 3.8+
**License:** MIT

---

## 1. Executive Summary

The **Vietnamese Fake News Detection System** is a research-grade, end-to-end framework designed to identify misinformation within the Vietnamese digital news ecosystem. Unlike traditional text-only approaches, this system implements a **Multimodal** architecture, treating both textual content and visual evidence (images) as primary signals for verification.

This project addresses a critical gap in current misinformation research: the lack of robust tools for low-resource languages like Vietnamese, where specific linguistic nuances and local web infrastructure challenges (legacy SSL, unstable connections) often break standard scraping and modeling pipelines.

By integrating a high-performance asynchronous crawler with the state-of-the-art **COOLANT** (Cross-modal Contrastive Learning) neural network, this system provides a complete lifecycle solution—from raw data acquisition to final veracity prediction.

---

## 2. Architectural Philosophy & Design Principles

### 2.1. The Challenge of Vietnamese Misinformation
Detecting fake news in Vietnam presents unique challenges that dictated our architectural choices:

*   **Linguistic Complexity:** Vietnamese is a tonal, analytic language. Standard tokenizers often fail to capture word boundaries correctly (e.g., "đất nước" is one word meaning "country", not two words "earth" and "water"). We intentionally integrate `UnderTheSea` and `PhoBERT` to handle these nuances natively.
*   **The "Multimodal Semantic Gap":** Fake news often pairs real text with misleading images (or vice versa). A system looking at text alone would miss the deception. Our architecture uses **Ambiguity Learning** (via Variational Autoencoders) to explicitly model the "distance" between what is said and what is shown.
*   **Infrastructure Fragility:** Many legitimate Vietnamese news sources, particularly local or government sites, operate on legacy infrastructure. This includes outdated SSL certificates (`UNSAFE_LEGACY_RENEGOTIATION_DISABLED`) and strict rate limits. Our crawler is built to be resilient, not just fast.

### 2.2. Core Pillars
1.  **Resilience First:** The crawler prefers finishing the job over speed. It implements exponential backoff, state preservation (resume capability), and specific SSL context overrides to ensure no URL is left behind due to technical debt on the target server.
2.  **Modularity:** The "Factory Pattern" is used throughout. Adding a new news source is as simple as adding a new class file; no core code modification is required.
3.  **Data-Centricity:** The system is designed around the `ViFactCheck` dataset structure, ensuring that all crawled data is immediately compatible with existing benchmarks.

---

## 3. System Architecture & Data Flow

### 3.1. High-Level Data Flow
The data travels through four distinct stages:

```plaintext
[ Stage 1: Acquisition ]
    |
    +-> Input: List of URLs (from ViFactCheck dataset)
    |
    +-> Crawler Engine (Async/Httpx)
        |-> Factory Router -> Selects specific Site Crawler
        |-> Fetch HTML -> Parse DOM -> Extract Text & Image URLs
        |-> Download Images -> Hash & Convert to JPG
    |
    +-> Output: Raw JSON Data + Raw Image Files

[ Stage 2: Preprocessing ]
    |
    +-> Text Pipeline
        |-> Regex Cleaning (Remove HTML, URLs, Special Chars)
        |-> Normalization (UnderTheSea)
        |-> Tokenization (PhoBERT)
    |
    +-> Image Pipeline
        |-> Resize (224x224)
        |-> Normalization (Mean/Std)
    |
    +-> Output: PyTorch Tensors

[ Stage 3: Modeling (COOLANT) ]
    |
    +-> Encoders: FastCNN (Text) + ResNet (Image)
    +-> Ambiguity Learning (VAE) -> Calculates KL Divergence
    +-> Attention Mechanism (SENet) -> Weights features
    +-> Fusion Layer -> Combines modalities
    |
    +-> Output: Probability Score (Real vs. Fake)

[ Stage 4: Storage ]
    |
    +-> JSON Logs
    +-> Model Checkpoints (.pth)
```

---

## 4. Component Reference: Deep Dive

### 4.1. The Crawler Engine (`src/crawler/`)

The crawler is the workhorse of the system. It is designed to be **Asynchronous** (using `asyncio`) and **Object-Oriented**.

#### The Factory Pattern (`CrawlerFactory`)
Instead of a giant `if/else` block, we use a Factory pattern to route URLs. The `CrawlerFactory` inspects the domain of an incoming URL and instantiates the correct crawler class.

*   **File:** `src/crawler/crawler_factory.py`
*   **Logic:**
    1.  Parse URL domain (e.g., `vnexpress.net`).
    2.  Lookup domain in `CRAWLER_MAPPING`.
    3.  Return specific crawler instance (e.g., `VnExpressCrawler()`).
    4.  If no match, log error and skip.

#### Site-Specific Implementation
Each news site has its own class inheriting from `BaseCrawler`. This enforces a strict contract: every crawler **must** define how to find the title, content, and images.

**Example: `src/crawler/news/real/VnExpressCrawler.py`**
```python
class VnExpressCrawler(BaseCrawler):
    def __init__(self) -> None:
        super().__init__(url="https://vnexpress.net/")

    @property
    def title_selector(self) -> SelectorType:
        # Precise CSS selector for the headline
        return {"css_selector": ["h1.title-detail"]}

    @property
    def content_selector(self) -> SelectorType:
        # Selects multiple potential content areas to be robust
        return {
            "css_selector": ["p.Normal", "p.description", "section#article_content > p"]
        }
    
    # ... image and link selectors defined similarly
```

#### Network Client (`Helpers/httpx_client.py`)
We wrap `httpx` to add robustness:
*   **Browser Mimicry:** Sets `User-Agent` headers to look like a real browser (Chrome/Firefox) to avoid basic anti-bot blocks.
*   **Retry Logic:** Implements a loop that catches `5xx` errors and retries with exponential backoff (wait 2s, then 4s, then 8s).
*   **Legacy SSL:** The `openssl.cnf` injection in `main.py` is critical here. Without it, requests to older Vietnamese government sites would fail immediately with `SSLError`.

### 4.2. Data Processing (`src/preprocessing/`)

#### Text Preprocessing (`TextPreprocessor`)
Cleaning Vietnamese text requires specific steps:
1.  **Regex Filters:**
    *   Removes URLs: `http\S+|www\S+`
    *   Removes Special Punctuation: `[.,;:!?""(){}\[\]\\/|`~@#$%^&*+=<>—–]`
    *   Removes Numbers: `\d+` (optional, based on config)
2.  **Normalization:** Uses `underthesea.text_normalize()` to fix accent issues (e.g., converting "òa" to "oà" consistently).
3.  **Tokenization:**
    *   Model: `vinai/phobert-base`
    *   Why? BERT is trained on English. PhoBERT is trained on massive Vietnamese datasets, understanding that "Hà Nội" is an entity, not two separate words.

### 4.3. The COOLANT Model (`src/models/coolant.py`)

This is the brain of the system. **COOLANT** stands for **C**ross-m**O**dal c**O**ntrastive **L**earning for **A**daptive **N**ews **T**hreat detection.

#### Key Components:
1.  **Ambiguity Learning (VAE):**
    *   Real news has high correlation between text and image. Fake news often has a disconnect (ambiguity).
    *   We use a Variational Autoencoder to learn the distribution of this relationship.
    *   **Metric:** Symmetric KL Divergence (`skl`).
    *   **Code:** `p_z1_given_text.log_prob(z1) - p_z2_given_image.log_prob(z1)`

2.  **Cross-Modal Correlation:**
    *   Computes a similarity matrix between every word embedding and every image region embedding.
    *   Uses `CrossModule4Batch` to project these into a unified correlation tensor.

3.  **Squeeze-and-Excitation (SE) Attention:**
    *   Adapts the weight of features on the fly. If the image is blurry or irrelevant, the SE block will down-weight the image channels and focus on the text channels.

#### Loss Function Strategy
The model minimizes a composite loss:
```python
total_loss = (
    classification_weight * classification_loss +  # Did we guess Real/Fake correctly?
    contrastive_weight * contrastive_loss +        # Are text/image pairs from the same article aligned?
    similarity_weight * similarity_loss            # Explicit similarity supervision
)
```

---

## 5. User Manual & Operations

### 5.1. Setup & Installation
1.  **Environment:**
    ```bash
    conda env create -f environment.yml
    conda activate fake-new-detection
    ```
2.  **Verify OpenSSL:**
    Ensure `openssl.cnf` is in the root directory. This file is required for the crawler to function correctly on all sites.

### 5.2. Running the System
Execute the main entry point:
```bash
python src/main.py
```

### 5.3. Understanding Crawl Modes
The system interacts with you via the CLI to determine how to handle existing data.

*   **`yes_all` (The "Nuke" Option)**
    *   **Behavior:** Deletes ALL cache files (`crawling_status_*.json`) and `failed_urls_*.json`.
    *   **When to use:** You are starting a fresh experiment and want zero contamination from previous runs.
    *   **Warning:** This cannot be undone.

*   **`no_all` (The "Resume" Option)**
    *   **Behavior:** Loads existing cache files. It calculates `Set(Target URLs) - Set(Crawled URLs)` and only processes the difference.
    *   **When to use:** Your internet cut out, or the script crashed after 5 hours. This picks up exactly where it left off.

*   **`manual` (The "Hybrid" Option)**
    *   **Behavior:** Prompts you for *each* dataset split (Train, Test, Dev) individually.
    *   **When to use:** You want to re-crawl the `dev` set for validation but keep the massive `train` set intact.

### 5.4. Interpreting Outputs
*   **Success Logs:** Check `logs/app.log`. Look for `[INFO] Saving images for https://...`.
*   **Failure Logs:** Check `data/caches/failed_urls_*.json`.
    *   `"reason": "404"` -> Article deleted.
    *   `"reason": "No crawler found"` -> We don't have a class for this domain yet.
*   **Data:**
    *   `data/json/news_data_vifactcheck_train.json`: The final dataset.
    *   `data/jpg/vn_express/`: The images. Files are named `vn_express_<hash>.jpg`.

---

## 6. Developer Guide

### 6.1. How to Add a New Crawler
Found a new news source in the dataset? Here is how to support it.

1.  **Inspect the Site:** Open a generic article on the site (e.g., `example.com`). Use Chrome DevTools to find the CSS selectors for:
    *   Title (`h1`)
    *   Content (`div.body` or `p`)
    *   Images (`figure img`)

2.  **Create Class:** Create `src/crawler/news/real/ExampleCrawler.py`.
    ```python
    from crawler.base_crawler import BaseCrawler
    
    class ExampleCrawler(BaseCrawler):
        title_selector = {"css_selector": ["h1.title"]}
        # ... implement other selectors ...
    ```

3.  **Register Class:** Open `src/crawler/crawler_factory.py`.
    ```python
    from .news.real.ExampleCrawler import ExampleCrawler
    
    CRAWLER_MAPPING = {
        # ...
        "example.com": ExampleCrawler,
    }
    ```

4.  **Test:** Add a URL from that site to a test list or run `agent_test.py`.

### 6.2. Running Tests
We use `pytest` for unit testing.
```bash
python -m pytest
```
*   **`test_scripts.py`**: Tests basic utility functions.
*   **`agent_test.py`**: Integration tests for specific crawler logic.

---

## 7. Troubleshooting & FAQ

### 7.1. SSL Errors
**Error:** `[SSL: UNSAFE_LEGACY_RENEGOTIATION_DISABLED]`
**Cause:** The target server uses an ancient OpenSSL version (common in government sites).
**Fix:** The system handles this automatically via `src/main.py` setting `OPENSSL_CONF`. If you see this, ensure `openssl.cnf` exists in the root.

### 7.2. "No Crawler Found"
**Error:** Many entries in `failed_urls.json` say "No crawler found".
**Cause:** The URL belongs to a domain we haven't implemented yet (e.g., a blog or a minor news site).
**Fix:** Follow the "How to Add a New Crawler" guide above to implement it, or ignore if the volume is low.

### 7.3. 403 Forbidden
**Error:** Crawler fails with status 403.
**Cause:** The site has detected you as a bot.
**Fix:**
1.  Check `src/helpers/httpx_client.py`.
2.  Update the `User-Agent` string.
3.  Increase the delay between requests (though `asyncio` makes this tricky, you may need to throttle the semaphore).

---

## 8. Future Roadmap

*   **Social Media Crawling:** Add support for crawling Facebook and YouTube links (requires distinct logic from standard HTML parsing).
*   **OCR Integration:** Integrate an OCR module (like Tesseract or VietOCR) to read text embedded *inside* images, which is a common vector for fake news.
*   **Graph Neural Networks:** Explore GNNs to model the propagation of news across social networks, adding a third modality (propagation structure) to the existing text+image model.
*   **Real-time API:** Wrap the trained model in a FastAPI service to allow real-time checking of URLs submitted by users.

---

**End of Documentation**