import json
import os
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
from .news.real.TienPhongCrawler import TienPhongCrawler

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
    "tienphong.vn": TienPhongCrawler,
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


async def crawl_and_save_all(urls: List[str], output_filename: str, cache_filename: str = "crawling_status.json"):
    all_results_data = []
    file_handler = FileHandler()
    completed_urls = set()

    # Load completed URLs from cache
    if os.path.exists(cache_filename):
        with open(cache_filename, 'r') as f:
            completed_urls = set(json.load(f))
        tqdm.write(f"Loaded {len(completed_urls)} completed URLs from cache.")

    # Filter out already crawled URLs
    urls_to_crawl = [url for url in urls if url not in completed_urls]
    tqdm.write(f"Found {len(urls_to_crawl)} new URLs to crawl.")

    with tqdm(total=len(urls_to_crawl), desc="Crawling URLs") as pbar:
        for url in urls_to_crawl:
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

                        # Update and save cache immediately
                        completed_urls.add(url)
                        with open(cache_filename, 'w') as f:
                            json.dump(list(completed_urls), f)

                    else:
                        tqdm.write(f"  Failed to crawl {url}: {result.error}")
            else:
                tqdm.write(f"--- No crawler found for {url} ---")
            pbar.update(1)

    if all_results_data:
        tqdm.write(f"\nAppending {len(all_results_data)} new results to {output_filename}...")
        # Load existing data and append new results
        existing_data = []
        if os.path.exists(output_filename):
            with open(output_filename, 'r') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    tqdm.write(f"Warning: Could not decode JSON from {output_filename}. Starting with a new file.")
        
        existing_data.extend(all_results_data)

        file_handler.write(
            format_name="json",
            data=existing_data,
            file_name=output_filename,
        )
        tqdm.write(f"--- All results saved to {output_filename} ---")
    else:
        tqdm.write("\n--- No new results to save. ---")
