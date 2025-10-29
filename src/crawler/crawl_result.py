from typing import List, Dict

class CrawlResult:
    def __init__(self, url, html, success=True, error=None):
        self.url = url
        self.html = html
        self.success = success
        self.error = error
        self.title = ""
        self.images: List[Dict[str, str]] = []
        self.links: List[str] = []
        self.contents: List[str] = []
        self.markdown = ""
        self.content_text = ""

    def to_markdown(self):
        return self.html
