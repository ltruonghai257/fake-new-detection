from crawler.base_crawler import BaseCrawler
from crawler.typings import SelectorType


class DanTriCrawler(BaseCrawler):
    def __init__(self) -> None:
        super().__init__(url="https://dantri.com.vn/")

    @property
    def title_selector(self) -> SelectorType:
        return {"css_selector": ["h1.title-page"]}

    @property
    def image_selector(self) -> SelectorType:
        return {
            "css_selector": ["figure"],
            "image_tag_selector": "img",
            "caption_tag_selector": "figcaption",
            "image_tag_attr": "data-src",
        }

    @property
    def link_selector(self) -> SelectorType:
        return {"css_selector": ["a"]}

    @property
    def content_selector(self) -> SelectorType:
        return {"css_selector": ["div.singular-content > p"]}

    @property
    def url_prefix(self) -> str:
        return "https://dantri.com.vn"
