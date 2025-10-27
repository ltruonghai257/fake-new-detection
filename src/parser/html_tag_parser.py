from lxml import html
from typing import Union, List

from crawler.typings import ImageType, SelectorType
from helpers.string_handle import StringHandler

class HTMLTagParser:
    def __init__(self, html_content: str):
        self.tree = html.fromstring(html_content)

    def get_title(self, selector: SelectorType) -> str:
        """
        Extracts the title from the HTML content based on the provided selector(s).
        :param selector: SelectorType object
        :return: title as a string
        Example:
            parser = HTMLTagParser(html_content)
            title = parser.get_title(selector={'css_selector': ['.title', 'h1.main-title']})
            print(f"Title: {title}")
        """
        selectors = selector.get('css_selector') or selector.get('xpath_selector')
        selector_type = 'cssselect' if selector.get('css_selector') else 'xpath'

        if isinstance(selectors, str):
            selectors = [selectors]
        
        for s in selectors:
            if selector_type == 'xpath':
                title = self.tree.xpath(s)
            elif selector_type == 'cssselect':
                title = self.tree.cssselect(s)
            else:
                raise ValueError("Invalid selector_type. Must be 'xpath' or 'cssselect'.")
            print(f"Debug: Trying selector '{s}' with type '{selector_type}', found title: {title}")
            if title:
                return title[0].text_content().strip()
        return None

    def get_images(self, selector: SelectorType) -> list:
        selectors = selector.get('css_selector') or selector.get('xpath_selector')
        selector_type = 'cssselect' if selector.get('css_selector') else 'xpath'

        if isinstance(selectors, str):
            selectors = [selectors]
            
        for s in selectors:
            if selector_type == 'xpath':
                images = [img.get('src') for img in self.tree.xpath(s)]
            elif selector_type == 'cssselect':
                images = [img.get('src') for img in self.tree.cssselect(s)]
            else:
                raise ValueError("Invalid selector_type. Must be 'xpath' or 'cssselect'.")
            if images:
                return images
        return []

    def get_images_and_captions(self, selector: SelectorType) -> List[ImageType]:
        """
        Extracts images and their captions from the HTML content based on the provided selector(s).
        :param selector: SelectorType object
        :return: list of ImageType
        Example:
            parser = HTMLTagParser(html_content)
            images_with_captions = parser.get_images_and_captions(selector={'xpath_selector': ['//figure', './/div[@class="image-container"]']})
            for image in images_with_captions:
                print(f"Image URL: {image.src}, Caption: {image.caption}")

        """
        selectors = selector.get('css_selector') or selector.get('xpath_selector')
        selector_type = 'cssselect' if selector.get('css_selector') else 'xpath'

        if isinstance(selectors, str):
            selectors = [selectors]

        images = []
        for s in selectors:
            if selector_type == 'xpath':
                elements = self.tree.xpath(s)
            elif selector_type == 'cssselect':
                elements = self.tree.cssselect(s)
            else:
                raise ValueError("Invalid selector_type. Must be 'xpath' or 'cssselect'.")
            print(f"Debug: Trying selector '{s}' with type '{selector_type}', found elements: {elements}")
            for element in elements:
                img_tag = element.find('.//img')
                print(f"Debug: Found img_tag: {img_tag.get('src')}")
                if img_tag is not None:
                    src = img_tag.get('src')
                    caption_tag = element.find('.//figcaption')
                    caption = caption_tag.text_content().strip() if caption_tag is not None else None
                    images.append(ImageType(src=src, caption=caption))
            if images:
                return images
        return []

    def get_links(self, selector: SelectorType) -> list:
        """
        Extracts hyperlinks from the HTML content based on the provided selector(s).
        :param selector: SelectorType object
        :return: list of hyperlinks
        Example:
            parser = HTMLTagParser(html_content)
            links = parser.get_links(selector={'css_selector': ['//nav//a', './/div[@class="footer-links"]//a']})
            for link in links:
                print(f"Link URL: {link}") 
        """
        selectors = selector.get('css_selector') or selector.get('xpath_selector')
        selector_type = 'cssselect' if selector.get('css_selector') else 'xpath'

        if isinstance(selectors, str):
            selectors = [selectors]

        for s in selectors:
            elements = []
            if selector_type == 'xpath':
                elements = self.tree.xpath(s)
            elif selector_type == 'cssselect':
                elements = self.tree.cssselect(s)
            else:
                raise ValueError("Invalid selector_type. Must be 'xpath' or 'cssselect'.")

            links = []
            for a in elements:
                href = a.get('href')
                if StringHandler.is_valid_url_path(href):
                    links.append(href)
            if links:
                return links
        return []

    def get_content(self, selector: SelectorType) -> list:
        selectors = selector.get('css_selector') or selector.get('xpath_selector')
        selector_type = 'cssselect' if selector.get('css_selector') else 'xpath'

        if isinstance(selectors, str):
            selectors = [selectors]
            
        for s in selectors:
            elements = []
            if selector_type == 'xpath':
                elements = self.tree.xpath(s)
            elif selector_type == 'cssselect':
                elements = self.tree.cssselect(s)
            else:
                raise ValueError("Invalid selector_type. Must be 'xpath' or 'cssselect'.")
            content = [element.text_content() for element in elements]
            if content:
                return content
        return []
