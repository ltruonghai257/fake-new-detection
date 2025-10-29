from typing import Literal, Optional, List, TypedDict, Union, Dict

ModeCrawlerType = Literal["simple", "deep"]


class BaseCrawlerRunner(TypedDict, total=False):
    mode: ModeCrawlerType
    save_to_file: bool


# class
FormatReturnCrawlerType = Literal["json", "text", "html", "markdown", "screenshot"]
ExtensionReturnCrawlerType = Literal[".json", ".txt", ".html", ".md", ".png", ".jpg"]


class ImageType(TypedDict, total=False):
    src: str
    caption: Optional[str]


class SelectorType(TypedDict, total=False):
    """
    Configuration for selecting elements on a page.
    Attributes:
        css_selector (Optional[Union[str, List[str]]]): A CSS selector string or a list of CSS selector strings.
        xpath_selector (Optional[Union[str, List[str]]]): An XPath selector string or a list of XPath selector strings.
        custom_selector (Optional[Union[str, List[str]]]): A custom selector string or logic.
        image_tag_selector (Optional[str]): A selector for the image tag within an image container.
        caption_tag_selector (Optional[str]): A selector for the caption tag within an image container.
        image_tag_attr (Optional[str]): The attribute of the image tag that contains the image source URL.
    """

    css_selector: Optional[Union[str, List[str]]]
    xpath_selector: Optional[Union[str, List[str]]]
    custom_selector: Optional[Union[str, List[str]]]
    image_tag_selector: Optional[str]
    caption_tag_selector: Optional[str]
    image_tag_attr: Optional[str]


class CrawlResultType(TypedDict, total=False):
    url: str
    html: str
    success: bool
    error: Optional[str]
    title: str
    images: List[Dict[str, str]]
    links: List[str]
    contents: List[str]
    markdown: str
    content_text: str
