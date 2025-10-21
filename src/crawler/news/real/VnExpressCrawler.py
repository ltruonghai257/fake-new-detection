from typing import Optional, Union, List
from lxml import html
import os
from src.helpers import StringHandler
from crawl4ai.deep_crawling import FilterChain, DomainFilter, ContentTypeFilter, URLPatternFilter

from src.crawler import BaseCrawler
from src.crawler.typings import ModeCrawlerType, ExtensionReturnCrawlerType


class VnExpressCrawler(BaseCrawler):
    def __init__(self) -> None:
        super().__init__(url="https://vnexpress.net/")
        self.schema = "VnExpress"

    @property
    def title_selector(self) -> Union[str, List[str]]:
        return [".//title", "h1.title-detail"]

    @property
    def image_selector(self) -> Union[str, List[str]]:
        return ["//img", "meta[property='og:image']/@content"]

    @property
    def link_selector(self) -> Union[str, List[str]]:
        return ["//a", "link[@rel='canonical']/@href"]

    @property
    def extraction_schema(self) -> dict:
        return {
            "name": "VnExpressArticle",
            "baseSelector": "article.item-news",
            "fields": [
                {"name": "headline", "selector": "h3.title-news > a", "type": "text"},
                {"name": "summary", "selector": "p.description > a", "type": "text"},
                {"name": "link", "selector": "h3.title-news > a", "type": "attribute", "attribute": "href"},
                {"name": "image", "selector": "div.thumb-art img", "type": "attribute", "attribute": "src"},
            ]
        }

    @property
    def content_selector(self) -> Union[str, List[str]]:
        return ["p.Normal", "div.fck_detail p"]

    @property
    def filter_chain(self) -> "FilterChain":
        return FilterChain([
            DomainFilter(
                allowed_domains=["vnexpress.net"],
            ),
            ContentTypeFilter(
                allowed_types=["text/html"]
            ),
            URLPatternFilter(patterns=["*.html"]),
        ])

    async def _process_deep_crawl_result(self, result):
        # 1. Save images and get paths
        image_paths = await self._save_images(result, ".jpg")
        
        # 2. Prepare JSON data
        json_data = {
            "title": result.metadata.get("title", ""),
            "url": result.url,
            "contents": [result.markdown.raw_markdown] if hasattr(result, "markdown") and hasattr(result.markdown, "raw_markdown") else [],
            "images": list(set(image_paths)),
            "links": list(set(result.links))
        }
        
        # 3. Save JSON file
        file_name = StringHandler.sanitize_filename(result.url) + ".json"
        self.file_handler.write("json", self.name, file_name, json_data)
        print(f"Extracted content from: {result.url}")