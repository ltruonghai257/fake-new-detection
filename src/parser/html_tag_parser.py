from lxml import html
from typing import Union, List

from src.helpers.string_handle import StringHandler

class HTMLTagParser:
    def __init__(self, html_content: str):
        self.tree = html.fromstring(html_content)

    def get_title(self, selector: Union[str, List[str]] = './/title') -> str:
        if isinstance(selector, str):
            selectors = [selector]
        else:
            selectors = selector
        
        for s in selectors:
            title = self.tree.findtext(s)
            if title:
                return title
        return None

    def get_images(self, selector: Union[str, List[str]] = '//img') -> list:
        if isinstance(selector, str):
            selectors = [selector]
        else:
            selectors = selector
            
        for s in selectors:
            images = [img.get('src') for img in self.tree.xpath(s)]
            if images:
                return images
        return []

    def get_links(self, selector: Union[str, List[str]] = '//a') -> list:
        if isinstance(selector, str):
            selectors = [selector]
        else:
            selectors = selector

        for s in selectors:
            links = []
            for a in self.tree.xpath(s):
                href = a.get('href')
                if StringHandler.is_valid_url_path(href):
                    links.append(href)
            if links:
                return links
        return []

    def get_content(self, selector: Union[str, List[str]]) -> list:
        if isinstance(selector, str):
            selectors = [selector]
        else:
            selectors = selector
            
        for s in selectors:
            content = [element.text_content() for element in self.tree.xpath(s)]
            if content:
                return content
        return []
