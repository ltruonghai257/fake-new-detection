from typing import Optional, List, Dict
from urllib.parse import urlparse
from tqdm.asyncio import tqdm

from .base_crawler import BaseCrawler
from .crawl_result import CrawlResult
from helpers.file_handler.file_handler import FileHandler

from .news.real.VnExpressCrawler import VnExpressCrawler
from .news.real.BaoChinhPhuCrawler import BaoChinhPhuCrawler
from .news.real.DanTriCrawler import DanTriCrawler
from .news.real.NguoiLaoDongCrawler import NguoiLaoDongCrawler
from .news.real.TuoiTreCrawler import TuoiTreCrawler
from .news.real.BaoTinTucCrawler import BaoTinTucCrawler
from .news.real.PhapLuatHcmCrawler import PhapLuatHcmCrawler
from .news.real.ThanhNienCrawler import ThanhNienCrawler

# To add a new crawler, import it here and add it to the mapping.
CRAWLER_MAPPING = {
    "vnexpress.net": VnExpressCrawler,
    "baochinhphu.vn": BaoChinhPhuCrawler,
    "dantri.com.vn": DanTriCrawler,
    "nld.com.vn": NguoiLaoDongCrawler,
    "tuoitre.vn": TuoiTreCrawler,
    "baotintuc.vn": BaoTinTucCrawler,
    "plo.vn": PhapLuatHcmCrawler,
    "thanhnien.vn": ThanhNienCrawler,
}


def get_crawler(url: str) -> Optional[BaseCrawler]:
    """
    Factory function to get the appropriate crawler for a given URL.

    Args:
        url: The URL to be crawled.

    Returns:
        An instance of the appropriate BaseCrawler subclass, or None if no
        matching crawler is found.
    """
    try:
        domain = urlparse(url).netloc
        # Find the crawler class that matches the domain
        for key, crawler_class in CRAWLER_MAPPING.items():
            if key in domain:
                return crawler_class()
    except Exception as e:
        tqdm.write(f"Error finding crawler for url {url}: {e}")
    
    return None


async def crawl_and_save_all(urls: List[str], output_filename: str):
    all_results_data = []
    file_handler = FileHandler()

    with tqdm(total=len(urls), desc="Crawling URLs") as pbar:
        for url in urls:
            pbar.set_description(f"Crawling {url}")
            crawler = get_crawler(url)
            if crawler:
                results = await crawler.arun(url=url, save_to_file=False) # arun returns List[CrawlResult]
                for result in results:
                    if result.success:
                        pbar.set_description(f"Saving images for {url}")
                        saved_images = await crawler._save_images(result, ".jpg")

                        prepared_data = {
                            "title": result.title,
                            "content": result.content_text,
                            "source_url": result.url,
                            "other_urls": result.links,
                            "images": saved_images,
                            "contents": result.contents,
                        }
                        all_results_data.append(prepared_data)
                    else:
                        tqdm.write(f"  Failed to crawl {url}: {result.error}")
            else:
                tqdm.write(f"--- No crawler found for {url} ---")
            pbar.update(1)

    if all_results_data:
        tqdm.write(f"\nSaving all results to {output_filename}...")
        file_handler.write(
            format_name="json",
            data=all_results_data,
            file_name=output_filename,
        )
        tqdm.write(f"--- All results saved to {output_filename} ---")
    else:
        tqdm.write("\n--- No results to save. ---")
