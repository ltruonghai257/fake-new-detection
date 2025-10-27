from abc import ABC, abstractmethod
import json
import os
import uuid

from os.path import join
from typing import Dict, List, Optional, Union
from urllib import parse
from bs4 import BeautifulSoup

from .typings import (
    ExtensionReturnCrawlerType,
    ModeCrawlerType,
    ImageType,
    SelectorType
)
from exceptions import URLFormatException
from helpers import StringHandler
from helpers.httpx_client import BaseClient
from helpers.file_handler.file_handler import FileHandler


class CrawlResult:
    def __init__(self, url, html, success=True, error=None):
        self.url = url
        self.html = html
        self.success = success
        self.error = error
        self.title = ""
        self.images = []
        self.links = []
        self.contents = []
        self.markdown = ""

    def to_markdown(self):
        return self.html


class BaseCrawler(ABC):
    schema_extra = None

    def __init__(
        self,
        url: str = "",
        name: Optional[str] = None,
    ) -> None:
        if url and not StringHandler.is_url(url):
            raise URLFormatException(url=url)
        self.url = url
        if not name:
            self.name = StringHandler.class_name_to_snake_case(self.__class__.__name__)
        else:
            self.name = name
        self.file_handler = FileHandler()
        self.client = BaseClient(base_url=url)

    async def arun(
        self,
        url: Optional[str] = None,
        mode: ModeCrawlerType = "simple",
        save_to_file: bool = True,
        max_depth: Optional[int] = 1,
        save_format: ExtensionReturnCrawlerType = ".json",
        deep_crawl_config: Optional[dict] = None,
    ):
        url_to_crawl = url or self.url
        if mode == "simple":
            results = [await self.simple_crawling(url=url_to_crawl)]
        else:  # mode == "deep"
            results = await self.deep_crawling(url=url_to_crawl, max_depth=max_depth)

        if results:
            processed_data = []
            for result in results:
                if result and result.html:
                    parsed_data = self._parse_html_content(result.html, result.url)
                    for key, value in parsed_data.items():
                        setattr(result, key, value)
                    
                    if save_to_file and result.images:
                        saved_images = await self._save_images(result, ".jpg") # Assuming jpg for now
                        result.images = saved_images
                    
                    processed_data.append(self._prepare_data_for_saving(result))

            if save_to_file:
                if save_format == ".json":
                    filtered_data = [data for data in processed_data if data['source_url'].startswith(self.url_prefix) and data['source_url'].endswith(self.url_suffix)]
                    file_name = f"{self.name}.json"
                    self.file_handler.write(save_format.strip('.'), self.name, file_name, filtered_data)
                else:
                    # For other formats, we save one file per result
                    for i, result in enumerate(results):
                        if result.success:
                            file_name = StringHandler.sanitize_filename(result.url) + save_format
                            await self.save_to_file(result, ext=save_format, file_name=file_name)

        print(f"Crawling completed for URL: {url_to_crawl} in {mode} mode. With Results: {results}")
        return results

    def _parse_html_content(self, html: str, base_url: str) -> dict:
        if not html:
            return {}
        soup = BeautifulSoup(html, "html.parser")

        title = soup.select_one(self.title_selector['css_selector'][0]).get_text(strip=True) if soup.select_one(self.title_selector['css_selector'][0]) else ""

        images = []
        for figure in soup.select(self.image_selector['css_selector'][0]):
            img_tag = figure.find('img')
            caption_tag = figure.find('figcaption')
            
            if img_tag and 'data-src' in img_tag.attrs:
                image_src = img_tag['data-src']
                caption = caption_tag.get_text(strip=True) if caption_tag else ""
                images.append({'src_url': image_src, 'caption': caption})

        links = []
        for a in soup.select(self.link_selector['css_selector'][0]):
            href = a.get('href')
            if href and href.startswith(self.url_prefix) and href.endswith(self.url_suffix):
                links.append(parse.urljoin(base_url, href))
        
        contents = [p.get_text(strip=True) for p in soup.select(self.content_selector['css_selector'][0])]

        return {
            "title": title,
            "images": images,
            "other_urls": links,
            "content": "\n".join(contents),
        }

    async def simple_crawling(self, url: Optional[str] = None):
        url_to_crawl = url or self.url
        try:
            response = await self.client.get(url_to_crawl)
            return CrawlResult(url=url_to_crawl, html=response.text)
        except httpx.RequestError as e:
            return CrawlResult(url=url_to_crawl, html="", success=False, error=str(e))

    async def deep_crawling(self, url: Optional[str] = None, max_depth: int = 2):
        url_to_crawl = url or self.url
        results = []
        queue = [(url_to_crawl, 0)]
        visited = {url_to_crawl}

        while queue:
            current_url, depth = queue.pop(0)

            if depth > max_depth:
                continue

            try:
                response = await self.client.get(current_url)
                result = CrawlResult(url=current_url, html=response.text)
                results.append(result)

                if depth < max_depth:
                    soup = BeautifulSoup(response.text, "html.parser")
                    for link in soup.select(self.link_selector['css_selector'][0]):
                        href = link.get("href")
                        if href:
                            absolute_url = parse.urljoin(current_url, href)
                            if parse.urlparse(absolute_url).netloc == parse.urlparse(url_to_crawl).netloc and absolute_url not in visited:
                                visited.add(absolute_url)
                                queue.append((absolute_url, depth + 1))
            except httpx.RequestError as e:
                results.append(CrawlResult(url=current_url, html="", success=False, error=str(e)))
        
        return results

    @property
    @abstractmethod
    def title_selector(self) -> SelectorType:
        pass

    @property
    @abstractmethod
    def image_selector(self) -> SelectorType:
        pass

    @property
    @abstractmethod
    def link_selector(self) -> SelectorType:
        pass

    @property
    @abstractmethod
    def url_prefix(self) -> str:
        pass

    @property
    def url_suffix(self) -> str:
        return ".html"
    


    async def _save_images(self, result, save_format: ExtensionReturnCrawlerType) -> List[Dict[str, str]]:
        saved_image_paths = []
        if hasattr(result, 'images') and result.images:
            format_name = save_format.strip('.')
            folder_path = os.path.join(self.file_handler.root_folder, format_name, self.name)
            self.file_handler.mkdir_if_not_exists(folder_path)

            for image in result.images:
                image_url = image.get('src_url')
                caption = image.get('caption')
                if not image_url or not isinstance(image_url, str) or image_url.startswith("data:"):
                    continue

                if not image_url.startswith('http'):
                    image_url = parse.urljoin(self.url, image_url)

                try:
                    response = await self.client.get(image_url)
                    image_data = response.content
                    
                    image_filename = os.path.basename(parse.urlparse(image_url).path)
                    if not image_filename:
                        image_filename = f"{uuid.uuid4()}{save_format}"

                    if not os.path.splitext(image_filename)[1]:
                        image_filename += save_format

                    self.file_handler.write(format_name, self.name, image_filename, image_data)
                    
                    saved_image_paths.append({
                        'folder_path': os.path.join(format_name, self.name, image_filename),
                        'src_url': image_url,
                        'caption': caption
                    })
                    
                except httpx.RequestError as e:
                    print(f"Error downloading image: {e}")
                except Exception as e:
                    print(f"An error occurred while saving the image: {e}")
        return saved_image_paths

    def _prepare_data_for_saving(self, result) -> Optional[Dict]:
        return {
            "title": result.title,
            "content": result.contents,
            "source_url": result.url,
            "other_urls": result.links,
            "images": result.images,
        }

    async def save_to_file(self, result, ext: ExtensionReturnCrawlerType = ".txt", file_name: Optional[str] = None):
        data_to_save = self._prepare_data_for_saving(result)
        if data_to_save:
            if not file_name:
                file_name = StringHandler.sanitize_filename(
                    StringHandler.class_name_to_snake_case(f"{self.name}{ext}")
                )
            self.file_handler.write(ext.strip('.'), self.name, file_name, data_to_save)

    def read_from_file(self, ext: ExtensionReturnCrawlerType = ".txt", file_name: Optional[str] = None) -> Dict:
        if not file_name:
            file_name = StringHandler.sanitize_filename(
                StringHandler.class_name_to_snake_case(f"{self.name}{ext}")
            )
        file_path = self.file_handler.generate_file_path(ext.strip('.'), self.name, file_name)
        return self.file_handler.read(file_path)