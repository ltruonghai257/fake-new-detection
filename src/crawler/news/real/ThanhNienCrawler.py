from crawler.base_crawler import BaseCrawler
from crawler.typings import SelectorType


class ThanhNienCrawler(BaseCrawler):
    def __init__(self) -> None:
        super().__init__(url="https://thanhnien.vn/")

    @property
    def title_selector(self) -> SelectorType:
        return {"css_selector": ["h1.detail-title"]}

    @property
    def image_selector(self) -> SelectorType:
        return {
            "css_selector": ["figure"],
            "image_tag_selector": "img",
            "caption_tag_selector": "figcaption",
            "image_tag_attr": "src",
        }

    @property
    def link_selector(self) -> SelectorType:
        return {"css_selector": ["a"]}

    @property
    def content_selector(self) -> SelectorType:
        return {"css_selector": ["div.detail-content > p"]}

    @property
    def url_prefix(self) -> str:
        return "https://thanhnien.vn"
