from os.path import join
from typing import TypedDict, Optional

from crawl4ai import BrowserConfig, CrawlerRunConfig, AsyncWebCrawler

from crawler.typings import (
    BrowserConfigType,
    CrawlerRunConfigType,
    ExtensionReturnCrawlerType,
)
from exceptions import URLFormatException
from helpers import StringHandler
from helpers.file_handler.file_handler import FileHandler


class BaseCrawler:
    def __init__(
        self,
        folder_path: str,
        url: str = "",
        ext: ExtensionReturnCrawlerType = ".json",
    ) -> None:
        if url and not StringHandler.is_url(url):
            raise URLFormatException(url=url)
        self.url = url
        self.browser_config = BrowserConfig()
        self.folder_path = folder_path
        self.crawler_run_config = CrawlerRunConfig()
        self.name = self.__class__.__name__
        self.file_path = self._generate_file_path(folder_path, ext)
        self.file_handler = FileHandler()

    def _generate_file_path(
        self, folder_path: str, extension: ExtensionReturnCrawlerType
    ) -> str:
        standard_file_name = StringHandler.sanitize_filename(
            StringHandler.class_name_to_snake_case(self.name)
        )
        return join(folder_path, f"{standard_file_name}{extension}")

    def config_browser(self, **kwargs: BrowserConfigType) -> None:
        self.browser_config = BrowserConfigType(**kwargs)

    def config_crawler_run(self, **kwargs: CrawlerRunConfigType) -> None:
        self.crawler_run_config = CrawlerRunConfig(**kwargs)

    async def simple_crawling(self, url: Optional[str] = None):
        if not url:
            url = self.url
        if not StringHandler.is_url(url):
            raise URLFormatException(url=url)
        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            result = await crawler.arun(url=url, config=self.crawler_run_config)
            return result

    def save_to_file(self, data: TypedDict) -> None:
        self.file_handler.write(self.file_path, data)

    def read_from_file(self):
        self.file_handler.read(self.file_path)
