from os.path import join
from typing import TypedDict, Optional

from crawl4ai import BrowserConfig, CrawlerRunConfig, AsyncWebCrawler

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
        if url and not StringHandler.is_url(url):
            raise URLFormatException(url=url)
        self.url = url
        self.browser_config = BrowserConfig()
        self.folder_path = folder_path
        self.crawler_run_config = CrawlerRunConfig()
        self.name = self.__class__.__name__ if not name else name
        self.file_handler = FileHandler()
        self.file_handler.mkdir_if_not_exists(folder_path)

    def _generate_file_path(
        self, folder_path: str, extension: ExtensionReturnCrawlerType
    ) -> str:
        standard_file_name = StringHandler.sanitize_filename(
            StringHandler.class_name_to_snake_case(f"{self.name}{extension}")
        )
        return join(folder_path, f"{standard_file_name}")

    def config_browser(self, **kwargs: BrowserConfigType) -> None:
        self.browser_config = BrowserConfig(**kwargs)

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

    # async def to_markdown(self, url: Optional[str] = None) -> str:
    def save_to_file(
        self, data: TypedDict, ext: ExtensionReturnCrawlerType = ".txt"
    ) -> None:
        file_path = self._generate_file_path(self.folder_path, ext)
        self.file_handler.write(file_path, data)

    def read_from_file(self, ext: ExtensionReturnCrawlerType = ".txt") -> TypedDict:
        file_path = self._generate_file_path(self.folder_path, ext)
        return self.file_handler.read(file_path)
