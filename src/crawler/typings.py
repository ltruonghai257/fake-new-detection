from typing import Literal, Optional, List
from typing import TypedDict, Union


class BrowserConfigType(TypedDict, total=False):
    """
    Configuration for the browser instance.
    Attributes:
        browser_type (str): Type of the browser (e.g., 'chromium', 'firefox', 'webkit').
        headless (bool): Whether to run the browser in headless mode. No visible UI.
        user_agent (str): The user agent string to be used by the browser.
        light_mode (bool): Whether to use light mode for the browser. If True, the browser will disable some background for performance gain.
    """

    browser_type: str
    headless: bool
    user_agent: str
    light_mode: bool


# A) Extraction & Markdown
class ExtractionConfig(TypedDict, total=False):
    word_count_threshold: int  # default ~200
    extraction_strategy: Optional[str]  # ExtractionStrategy
    markdown_generator: Optional[str]  # MarkdownGenerationStrategy

    css_selector: Optional[str]
    target_elements: Optional[List[str]]

    excluded_tags: Optional[List[str]]
    excluded_selector: Optional[str]

    only_text: bool
    prettify: bool
    keep_data_attributes: bool
    remove_forms: bool


# B) Cache
class CacheConfig(TypedDict, total=False):
    cache_mode: Optional[str]  # CacheMode (ENABLED, BYPASS, DISABLED, etc.)
    session_id: Optional[str]

    bypass_cache: bool
    disable_cache: bool
    no_cache_read: bool
    no_cache_write: bool


# C) Page Navigation & Timing
class NavigationConfig(TypedDict, total=False):
    wait_until: str  # e.g. "networkidle" or "domcontentloaded"
    page_timeout: int  # default 60000 ms
    wait_for: Optional[str]  # CSS selector or JS condition
    wait_for_images: bool
    delay_before_return_html: float  # default 0.1
    check_robots_txt: bool
    mean_delay: float  # default 0.1
    max_range: float  # default 0.3
    semaphore_count: int  # default 5


# F) Link Filtering
class LinkFilterConfig(TypedDict, total=False):
    exclude_social_media_domains: Optional[List[str]]
    exclude_external_links: bool
    exclude_social_media_links: bool
    exclude_domains: Optional[List[str]]


# G) Debug & Logging
class DebugConfig(TypedDict, total=False):
    verbose: bool
    log_console: bool


# H) Virtual Scroll
class VirtualScrollConfig(TypedDict, total=False):
    virtual_scroll_config: Optional[
        Union[dict, "VirtualScrollConfig"]
    ]  # nested config allowed


# --- Final combined config ---
class CrawlerRunConfigType(
    ExtractionConfig,
    CacheConfig,
    NavigationConfig,
    LinkFilterConfig,
    DebugConfig,
    VirtualScrollConfig,
    total=False,
):
    """Final crawler config type combining all sections"""

    pass


ModeCrawlerType = Literal["simple", "deep"]


class BaseCrawlerRunner(TypedDict, total=False):
    mode: ModeCrawlerType
    save_to_file: bool


# class
FormatReturnCrawlerType = Literal["json", "text", "html", "markdown", "screenshot"]
ExtensionReturnCrawlerType = Literal[".json", ".txt", ".html", ".md", ".png"]
