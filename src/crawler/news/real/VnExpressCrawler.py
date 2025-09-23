from src.crawler import BaseCrawler


class VnExpressCrawler(BaseCrawler):
    def __init__(self, folder_path: str, url: str = "") -> None:
        super().__init__(folder_path=folder_path, url="https://vnexpress.net/")
