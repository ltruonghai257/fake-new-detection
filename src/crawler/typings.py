from typing import Literal, Optional, List, TypedDict, Union


ModeCrawlerType = Literal["simple", "deep"]


class BaseCrawlerRunner(TypedDict, total=False):
    mode: ModeCrawlerType
    save_to_file: bool


# class
FormatReturnCrawlerType = Literal["json", "text", "html", "markdown", "screenshot"]
ExtensionReturnCrawlerType = Literal[".json", ".txt", ".html", ".md", ".png"]

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
    """
    css_selector: Optional[Union[str, List[str]]]
    xpath_selector: Optional[Union[str, List[str]]]
    custom_selector: Optional[Union[str, List[str]]]