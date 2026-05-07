from crawler.base_crawler import BaseCrawler
from crawler.typings import SelectorType


class TienPhongCrawler(BaseCrawler):
    def __init__(self) -> None:
        super().__init__(url="https://tienphong.vn/")

    @property
    def title_selector(self) -> SelectorType:
        # Placeholder: Please inspect the website and update this selector
        return {"css_selector": ["h1.article__title"]}

    @property
    def image_selector(self) -> SelectorType:
        # TienPhong uses <table class="picture"> with <td class="caption"> <p>
        return {
            "css_selector": ["table.picture", "figure"],
            "image_tag_selector": "img",
            "caption_tag_selector": "td.caption",
            "image_tag_attr": "src",
        }

    @property
    def link_selector(self) -> SelectorType:
        return {"css_selector": ["a"]}

    @property
    def content_selector(self) -> SelectorType:
        # Placeholder: Please inspect the website and update this selector
        return {"css_selector": ["div.article__body > p"]}

    @property
    def url_prefix(self) -> str:
        return "https://tienphong.vn"
