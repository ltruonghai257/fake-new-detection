from os.path import join
from typing import TypedDict, Optional
from urllib import parse as urllib

from crawl4ai import BrowserConfig, CrawlerRunConfig, AsyncWebCrawler
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy

from .typings import (
    BrowserConfigType,
    CrawlerRunConfigType,
    ExtensionReturnCrawlerType,
)
from ..exceptions import URLFormatException
from ..helpers import StringHandler
from ..helpers.file_handler.file_handler import FileHandler


class BaseCrawler:
    def __init__(
        self,
        folder_path: str,
        url: str = "",
        name: Optional[str] = None,
    ) -> None:
        self.browser_config = None
        self.crawler_run_config = None
        if url and not StringHandler.is_url(url):
            raise URLFormatException(url=url)
        self.url = url
        self.folder_path = folder_path
        self.name = self.__class__.__name__ if not name else name
        self.file_handler = FileHandler()
        self.file_handler.mkdir_if_not_exists(folder_path)
        self.config_browser()
        self.config_crawler_run()

    def _generate_file_path(
        self, folder_path: str, extension: ExtensionReturnCrawlerType
    ) -> str:
        standard_file_name = StringHandler.sanitize_filename(
            StringHandler.class_name_to_snake_case(f"{self.name}{extension}")
        )
        return join(folder_path, f"{standard_file_name}")

    def config_browser(
        self, config: Optional[BrowserConfigType] = None, can_return: bool = True
    ) -> None:
        self.browser_config = BrowserConfig(**config) if config else BrowserConfig()

    def config_crawler_run(self, config: Optional[CrawlerRunConfigType] = None) -> None:
        self.crawler_run_config = (
            CrawlerRunConfig(**config) if config else CrawlerRunConfig()
        )

    async def deep_crawling(
        self, url: Optional[str] = None, max_depth: Optional[int] = 2
    ):

        self.config_crawler_run(
            CrawlerRunConfigType(
                deep_crawl_strategy=BFSDeepCrawlStrategy(max_depth=max_depth),
                scraping_strategy=LXMLWebScrapingStrategy(),
                verbose=True,
            )
        )

        return await self.simple_crawling(url=url, is_deep=True)

    async def simple_crawling(self, url: Optional[str] = None, is_deep: bool = False):
        if not url:
            url = self.url
        else:
            url = urllib.urljoin(self.url, url)
        if not StringHandler.is_url(url):
            raise URLFormatException(url=url)

        async with AsyncWebCrawler(
            config=self.browser_config if not is_deep else None
        ) as crawler:
            result = await crawler.arun(url=url, config=self.crawler_run_config)
            return result

    # async def to_markdown(self, url: Optional[str] = None) -> str:
    def save_to_file(
        self, data: TypedDict, ext: ExtensionReturnCrawlerType = ".txt"
    ) -> None:
        file_path = self._generate_file_path(self.folder_path, ext)
        self.file_handler.write(file_path, data)

    def read_from_file(self, ext: ExtensionReturnCrawlerType = ".txt") -> TypedDict:
        file_path = self._generate_file_path(self.folder_path, ext)
        return self.file_handler.read(file_path)
