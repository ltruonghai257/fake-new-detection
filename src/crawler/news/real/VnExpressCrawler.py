from crawler.base import BaseCrawler
from crawler.typings import SelectorType


class VnExpressCrawler(BaseCrawler):
    def __init__(self) -> None:
        super().__init__(url="https://vnexpress.net/")

    @property
    def title_selector(self) -> SelectorType:
        return {"css_selector": ["h1.title-detail"]}

    @property
    def image_selector(self) -> SelectorType:
        return {"css_selector": ["figure.tplCaption"]}

    @property
    def link_selector(self) -> SelectorType:
        return {"css_selector": ["a"]}

    @property
    def content_selector(self) -> SelectorType:
        return {"css_selector": ["p.Normal"]}

    @property
    def url_prefix(self) -> str:
        return "https://vnexpress.net"
