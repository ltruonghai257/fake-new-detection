import hashlib
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Union
from urllib import parse
from PIL import Image
import io

import httpx
from bs4 import BeautifulSoup, Tag
from lxml import html

from exceptions import URLFormatException
from helpers import StringHandler
from helpers.file_handler.file_handler import FileHandler
from helpers.httpx_client import BaseClient
from .crawl_result import CrawlResult
from .typings import ExtensionReturnCrawlerType, SelectorType


class BaseCrawler(ABC):
    schema_extra = None

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

    @property
    @abstractmethod
    def content_selector(self) -> SelectorType:
        pass

    def __init__(
        self,
        url: str = "",
        name: Optional[str] = None,
    ) -> None:
        if url and not StringHandler.is_url(url):
            raise URLFormatException(url=url)
        self.url = url
        if not name:
            class_name = self.__class__.__name__
            if class_name.endswith("Crawler"):
                class_name = class_name[:-7]  # Remove "Crawler"
            self.name = StringHandler.class_name_to_snake_case(class_name)
        else:
            self.name = name
        self.file_handler = FileHandler()
        self.client = BaseClient(base_url=url, verify=False)

    async def arun(
        self,
        url: Optional[Union[str, List[str]]] = None,
        save_to_file: bool = True,
        max_depth: Optional[int] = 1,
        save_format: ExtensionReturnCrawlerType = ".json",
        output_filename: Optional[str] = None,
    ) -> List[CrawlResult]:
        url_to_crawl = url or self.url
        results: List[CrawlResult] = []

        if isinstance(url_to_crawl, str):
            results = await self.simple_crawling(url=url_to_crawl)
        elif isinstance(url_to_crawl, list):
            results = await self.simple_crawling(url=url_to_crawl)

        if results:
            processed_data = []
            for result in results:
                if result and result.html:
                    parsed_data = self._parse_html_content(result.html, result.url)
                    for key, value in parsed_data.items():
                        setattr(result, key, value)

                    if save_to_file and result.images:
                        saved_images = await self._save_images(
                            result, ".jpg"
                        )  # Assuming jpg for now
                        result.images = saved_images

                    processed_data.append(self._prepare_data_for_saving(result))

            if save_to_file:
                if save_format == ".json":
                    file_name = output_filename or f"{self.name}.json"
                    self.file_handler.write(
                        format_name=save_format.strip("."),
                        data=processed_data,
                        file_name=file_name,
                    )
                else:
                    # For other formats, we save one file per result
                    for i, result in enumerate(results):
                        if result.success:
                            file_name = (
                                StringHandler.sanitize_filename(result.url)
                                + save_format
                            )
                            await self.save_to_file(
                                result, ext=save_format, file_name=file_name
                            )

        return results

    def _get_elements(
        self,
        soup: BeautifulSoup,
        lxml_tree: html.HtmlElement,
        selector: SelectorType,
        find_all: bool = True,
    ) -> List:
        elements = []
        if "css_selector" in selector:
            for css_selector in selector["css_selector"]:
                if find_all:
                    elements = soup.select(css_selector)
                else:
                    elements = [soup.select_one(css_selector)]
                if elements and elements[0] is not None:
                    return elements
        if "xpath_selector" in selector:
            for xpath_selector in selector["xpath_selector"]:
                elements = lxml_tree.xpath(xpath_selector)
                if elements:
                    return elements
        return []

    def _parse_html_content(self, html_content: str, base_url: str) -> dict:
        if not html_content:
            return {}
        soup = BeautifulSoup(html_content, "html.parser")
        lxml_tree = html.fromstring(html_content)

        title_element = self._get_elements(
            soup, lxml_tree, self.title_selector, find_all=False
        )
        title = title_element[0].get_text(strip=True) if title_element else ""

        images = []
        image_elements = self._get_elements(soup, lxml_tree, self.image_selector)
        image_tag_selector = self.image_selector.get("image_tag_selector", "img")
        caption_tag_selector = self.image_selector.get(
            "caption_tag_selector", "figcaption"
        )
        image_tag_attr = self.image_selector.get("image_tag_attr", "src")

        for figure in image_elements:
            img_tag = figure.find(image_tag_selector)
            caption_tag = figure.find(caption_tag_selector)

            if img_tag and image_tag_attr in img_tag.attrs:
                image_src = img_tag[image_tag_attr]
                caption = caption_tag.get_text(strip=True) if caption_tag else ""
                images.append({"src_url": image_src, "caption": caption})

        links = []
        link_elements = self._get_elements(soup, lxml_tree, self.link_selector)
        for a in link_elements:
            href = a.get("href") if isinstance(a, Tag) else a.get("href")
            if (
                href
                and href.startswith(self.url_prefix)
                and href.endswith(self.url_suffix)
            ):
                links.append(parse.urljoin(base_url, href))

        contents = []
        content_elements = self._get_elements(soup, lxml_tree, self.content_selector)
        for p in content_elements:
            contents.append(p.get_text(strip=True))

        return {
            "title": title,
            "images": images,
            "links": links,
            "content_text": "\n".join(contents),
            "contents": contents,
        }

    async def _execute_single_crawl(self, url: str) -> CrawlResult:
        try:
            response = await self.client.get(url)
            return CrawlResult(url=url, html=response.text)
        except httpx.RequestError as e:
            return CrawlResult(url=url, html="", success=False, error=str(e))

    async def simple_crawling(self, url: Union[str, List[str]]) -> List[CrawlResult]:
        if isinstance(url, str):
            return [await self._execute_single_crawl(url)]
        elif isinstance(url, list):
            results = []
            for u in url:
                results.append(await self._execute_single_crawl(u))
            return results
        return []  # Should not happen with type hints

    async def _recursive_crawl(
        self,
        url: str,
        depth: int,
        max_depth: int,
        visited: Set[str],
        results: List[CrawlResult],
    ):
        if depth > max_depth or url in visited:
            return

        visited.add(url)

        try:
            response = await self.client.get(url)
            result = CrawlResult(url=url, html=response.text)
            results.append(result)

            if depth < max_depth:
                soup = BeautifulSoup(response.text, "html.parser")
                lxml_tree = html.fromstring(response.text)
                link_elements = self._get_elements(soup, lxml_tree, self.link_selector)
                for link in link_elements:
                    href = (
                        link.get("href") if isinstance(link, Tag) else link.get("href")
                    )
                    if href:
                        absolute_url = parse.urljoin(url, href)
                        if (
                            parse.urlparse(absolute_url).netloc
                            == parse.urlparse(self.url).netloc
                        ):
                            await self._recursive_crawl(
                                absolute_url, depth + 1, max_depth, visited, results
                            )
        except httpx.RequestError as e:
            results.append(CrawlResult(url=url, html="", success=False, error=str(e)))

    async def deep_crawling(
        self, url: Optional[str] = None, max_depth: int = 2
    ) -> List[CrawlResult]:
        url_to_crawl = url or self.url
        results = []
        visited = set()
        await self._recursive_crawl(url_to_crawl, 0, max_depth, visited, results)
        return results

    async def _save_images(
        self, result, save_format: ExtensionReturnCrawlerType
    ) -> List[Dict[str, str]]:
        saved_image_paths = []
        if hasattr(result, "images") and result.images:
            format_name = save_format.strip(".")

            for image in result.images:
                image_url = image.get("src_url")
                caption = image.get("caption")
                if (
                    not image_url
                    or not isinstance(image_url, str)
                    or image_url.startswith("data:")
                ):
                    continue

                if not image_url.startswith("http"):
                    image_url = parse.urljoin(self.url, image_url)

                try:
                    response = await self.client.get(image_url)
                    image_data = response.content

                    # Open the image and convert to JPG
                    img = Image.open(io.BytesIO(image_data))
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    output_buffer = io.BytesIO()
                    img.save(output_buffer, format="JPEG")
                    image_data = output_buffer.getvalue()

                    hash_id = hashlib.sha256(image_url.encode()).hexdigest()[:10]
                    image_filename = f"{self.name}_{hash_id}.jpg"

                    self.file_handler.write(
                        format_name=format_name,
                        data=image_data,
                        class_name=self.name,
                        file_name=image_filename,
                    )

                    saved_image_paths.append(
                        {
                            "folder_path": os.path.join(
                                format_name, self.name, image_filename
                            ),
                            "src_url": image_url,
                            "caption": caption,
                        }
                    )

                except httpx.RequestError as e:
                    print(f"Error downloading image: {e}")
                except Exception as e:
                    print(f"An error occurred while saving the image: {e}")
        return saved_image_paths

    def _prepare_data_for_saving(self, result: CrawlResult) -> Optional[Dict]:
        """Prepare data for saving using the configured output format"""
        from .output_formats import OutputFormatter
        
        # Get the format name from schema_extra or use default
        format_name = getattr(self, 'output_format', 'default')
        formatter = OutputFormatter.get_formatter(format_name)
        return formatter(result)

    async def save_to_file(
        self,
        result,
        ext: ExtensionReturnCrawlerType = ".txt",
        file_name: Optional[str] = None,
    ):
        data_to_save = self._prepare_data_for_saving(result)
        if data_to_save:
            if not file_name:
                file_name = StringHandler.sanitize_filename(
                    StringHandler.class_name_to_snake_case(f"{self.name}{ext}")
                )
            self.file_handler.write(
                format_name=ext.strip("."),
                data=data_to_save,
                class_name=self.name,
                file_name=file_name,
            )

    def read_from_file(
        self, ext: ExtensionReturnCrawlerType = ".txt", file_name: Optional[str] = None
    ) -> Dict:
        if not file_name:
            file_name = StringHandler.sanitize_filename(
                StringHandler.class_name_to_snake_case(f"{self.name}{ext}")
            )
        file_path = self.file_handler.generate_file_path(
            ext.strip("."), self.name, file_name
        )
        return self.file_handler.read(file_path)
