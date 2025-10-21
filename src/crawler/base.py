from abc import ABC, abstractmethod
import json
import os
import uuid
import requests
from os.path import join
from typing import Dict, List, Optional, Union
from urllib import parse

from crawl4ai import BrowserConfig, CrawlerRunConfig, AsyncWebCrawler
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy, FilterChain

from .typings import (
    BrowserConfigType,
    CrawlerRunConfigType,
    ExtensionReturnCrawlerType,
    ModeCrawlerType,
)
from ..exceptions import URLFormatException
from ..helpers import StringHandler
from ..helpers.file_handler.file_handler import FileHandler
from ..parser.html_tag_parser import HTMLTagParser


class BaseCrawler(ABC):
    schema_extra = None

    def __init__(
        self,
        url: str = "",
        name: Optional[str] = None,
    ) -> None:
        self.browser_config = None
        self.crawler_run_config = None
        if url and not StringHandler.is_url(url):
            raise URLFormatException(url=url)
        self.url = url
        if not name:
            self.name = StringHandler.class_name_to_snake_case(self.__class__.__name__)
        else:
            self.name = name
        self.file_handler = FileHandler()
        self.config_browser()
        self.config_crawler_run()

    async def arun(
        self,
        url: Optional[str] = None,
        mode: ModeCrawlerType = "simple",
        save_to_file: bool = True,
        max_depth: Optional[int] = 1,
        save_format: ExtensionReturnCrawlerType = ".md",
        deep_crawl_config: Optional[dict] = None,
    ):
        url_to_crawl = url or self.url
        if mode == "simple":
            results = [await self.simple_crawling(url=url_to_crawl)]
        else: # mode == "deep"
            results = await self.deep_crawling(url=url_to_crawl, max_depth=max_depth, deep_crawl_config=deep_crawl_config)

        if results:
            processed_results = []
            for result in results:
                if result:
                    self._parse_html_content(result)
                    processed_results.append(result)

            if save_to_file:
                if save_format == ".json":
                    all_data = [self._prepare_data_for_saving(res, save_format) for res in processed_results]
                    file_name = StringHandler.sanitize_filename(
                        StringHandler.class_name_to_snake_case(f"{self.name}{save_format}")
                    )
                    self.file_handler.write(save_format.strip('.'), self.name, file_name, all_data)
                else:
                    for res in processed_results:
                        await self.save_to_file(res, ext=save_format)

        return results

    def _parse_html_content(self, result):
        if result and result.html:
            parser = HTMLTagParser(result.html)
            result.title = parser.get_title(selector=self.title_selector)
            result.images = parser.get_images(selector=self.image_selector)
            result.links = parser.get_links(selector=self.link_selector)
            result.contents = parser.get_content(selector=self.content_selector)

    async def _process_result(self, result, save_to_file: bool, save_format: ExtensionReturnCrawlerType):
        self._parse_html_content(result)
        if save_to_file:
            await self.save_to_file(result, ext=save_format)

    @property
    @abstractmethod
    def title_selector(self) -> Union[str, List[str]]:
        pass

    @property
    @abstractmethod
    def image_selector(self) -> Union[str, List[str]]:
        pass

    @property
    @abstractmethod
    def link_selector(self) -> Union[str, List[str]]:
        pass

    @property
    @abstractmethod
    def content_selector(self) -> Union[str, List[str]]:
        pass

    @property
    @abstractmethod
    def extraction_schema(self) -> dict:
        pass

    @property
    @abstractmethod
    def filter_chain(self) -> "FilterChain":
        pass

    @abstractmethod
    async def _process_deep_crawl_result(self, result):
        pass



    async def extract_with_schema(self, url: Optional[str] = None, config: Optional[CrawlerRunConfigType] = None):
        if not config:
            config = {}
            
        config['extraction_strategy'] = JsonCssExtractionStrategy(self.extraction_schema)
        
        self.config_crawler_run(config)
        
        url_to_crawl = url or self.url
        result = await self.simple_crawling(url=url_to_crawl)
        
        if result and result.success and hasattr(result, 'extracted_content') and result.extracted_content:
            try:
                return json.loads(result.extracted_content)
            except json.JSONDecodeError:
                return None
        return None

    async def _save_images(self, result, save_format: ExtensionReturnCrawlerType) -> List[str]:
        saved_image_paths = []
        if hasattr(result, 'images') and result.images:
            format_name = save_format.strip('.')
            
            for image_url in result.images:
                if not image_url or not isinstance(image_url, str) or image_url.startswith("data:"):
                    continue

                # Make sure the URL is absolute
                if not image_url.startswith('http'):
                    image_url = parse.urljoin(self.url, image_url)

                try:
                    # Download the image
                    response = requests.get(image_url)
                    response.raise_for_status()
                    image_data = response.content
                    
                    # Create a file path for the image
                    image_filename = os.path.basename(parse.urlparse(image_url).path)
                    if not image_filename:
                        # create a random name
                        image_filename = f"{uuid.uuid4()}{save_format}"

                    # Add extension if missing
                    if not os.path.splitext(image_filename)[1]:
                        image_filename += save_format

                    self.file_handler.write(format_name, self.name, image_filename, image_data)
                    image_path = self.file_handler.generate_file_path(format_name, self.name, image_filename)
                    saved_image_paths.append(image_path)
                    
                except requests.exceptions.RequestException as e:
                    print(f"Error downloading image: {e}")
                except Exception as e:
                    print(f"An error occurred while saving the image: {e}")
        return saved_image_paths

    def _prepare_data_for_saving(self, result, save_format: ExtensionReturnCrawlerType) -> Optional[Dict]:
        if save_format == ".md" and hasattr(result, "to_markdown"):
            return {"content": result.to_markdown()}
        elif save_format == ".json":
            return {
                "title": result.title,
                "url": result.url,
                "links": result.links,
                "images": result.images,
                "contents": result.contents,
                "html": result.html,
            }
        elif save_format == ".txt":
            if hasattr(result, "markdown") and hasattr(result.markdown, "raw_markdown"):
                return {"content": result.markdown.raw_markdown}
            else:
                return {"content": ""}
        return None

    async def save_to_file(self, result, ext: ExtensionReturnCrawlerType = ".txt", file_name: Optional[str] = None):
        if ext in (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg"):
            await self._save_images(result, ext)
        else:
            data_to_save = self._prepare_data_for_saving(result, ext)
            if data_to_save:
                if not file_name:
                    file_name = StringHandler.sanitize_filename(
                        StringHandler.class_name_to_snake_case(f"{self.name}{ext}")
                    )
                self.file_handler.write(ext.strip('.'), self.name, file_name, data_to_save)

    def config_browser(
        self, config: Optional[BrowserConfigType] = None
    ) -> None:
        self.browser_config = BrowserConfig(**config) if config else BrowserConfig()

    def config_crawler_run(self, config: Optional[CrawlerRunConfigType] = None) -> None:
        self.crawler_run_config = (
            CrawlerRunConfig(**config) if config else CrawlerRunConfig()
        )

    async def deep_crawling(self, url: Optional[str] = None, max_depth: int = 2, verbose: bool = False, deep_crawl_config: Optional[dict] = None):
        url_to_crawl = url or self.url
        
        if deep_crawl_config:
            strategy = BFSDeepCrawlStrategy(
                max_depth=deep_crawl_config.get("max_depth", 2),
                include_external=deep_crawl_config.get("include_external", False),
                max_pages=deep_crawl_config.get("max_pages", 50),
                filter_chain=self.filter_chain
            )
            config = {
                "deep_crawl_strategy": strategy,
                "stream": False,
                "verbose": True,
            }
            self.config_crawler_run(config)
        else:
            self.config_crawler_run(
                CrawlerRunConfigType(
                    deep_crawl_strategy=BFSDeepCrawlStrategy(max_depth=max_depth),
                    scraping_strategy=LXMLWebScrapingStrategy(),
                    verbose=verbose,
                )
            )

        return await self._execute_crawl(url=url_to_crawl, is_deep=True)

    async def simple_crawling(self, url: Optional[str] = None):
        url_to_crawl = url or self.url
        return await self._execute_crawl(url=url_to_crawl)

    async def _execute_crawl(self, url: str, is_deep: bool = False):
        # Ensure the URL is absolute
        final_url = parse.urljoin(self.url, url)
        if not StringHandler.is_url(final_url):
            raise URLFormatException(url=final_url)
        
        async with AsyncWebCrawler(
            config=self.browser_config if not is_deep else None
        ) as crawler:
            result = await crawler.arun(url=final_url, config=self.crawler_run_config)
            return result

    def read_from_file(self, ext: ExtensionReturnCrawlerType = ".txt", file_name: Optional[str] = None) -> Dict:
        if not file_name:
            file_name = StringHandler.sanitize_filename(
                StringHandler.class_name_to_snake_case(f"{self.name}{ext}")
            )
        file_path = self.file_handler.generate_file_path(ext.strip('.'), self.name, file_name)
        return self.file_handler.read(file_path)
