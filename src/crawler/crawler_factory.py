import json
import os
import time
from typing import Optional, List, Dict
from urllib.parse import urlparse
from tqdm.asyncio import tqdm
from datetime import datetime

from .base_crawler import BaseCrawler
from .crawl_result import CrawlResult
from helpers.file_handler.file_handler import FileHandler
from .output_formats import OutputFormatter
from helpers.logger import logger

from .news.real.VnExpressCrawler import VnExpressCrawler
from .news.real.BaoChinhPhuCrawler import BaoChinhPhuCrawler
from .news.real.DanTriCrawler import DanTriCrawler
from .news.real.NguoiLaoDongCrawler import NguoiLaoDongCrawler
from .news.real.TuoiTreCrawler import TuoiTreCrawler
from .news.real.BaoTinTucCrawler import BaoTinTucCrawler
from .news.real.PhapLuatHcmCrawler import PhapLuatHcmCrawler
from .news.real.ThanhNienCrawler import ThanhNienCrawler
from .news.real.TienPhongCrawler import TienPhongCrawler

class CrawlerFactory:
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

    def __init__(self, cache_filename: str = "crawling_status.json", failed_log_filename: str = "failed_urls.json"):
        self.cache_filename = cache_filename
        self.failed_log_filename = failed_log_filename

    def get_crawler(self, url: str) -> Optional[BaseCrawler]:
        """
        Factory method to get the appropriate crawler for a given URL.

        Args:
            url: The URL to be crawled.

        Returns:
            An instance of the appropriate BaseCrawler subclass, or None if no
            matching crawler is found.
        """
        try:
            domain = urlparse(url).netloc
            for key, crawler_class in self.CRAWLER_MAPPING.items():
                if key in domain:
                    return crawler_class()
        except Exception as e:
            logger.error(f"Error finding crawler for url {url}: {e}")
        
        return None

    def check_cache_file_exists(self) -> bool:
        """
        Check if the cache file exists.
        Returns:
            bool: True if the cache file exists, False otherwise.
        """
        return os.path.exists(self.cache_filename)

    def clear_cache(self):
        """
        Deletes the cache file if it exists.
        """
        if os.path.exists(self.cache_filename):
            os.remove(self.cache_filename)
            logger.info(f"--- Cache file '{self.cache_filename}' cleared. ---")

    async def crawl_and_save_all(self, urls: List[str], output_filename: str, format_name: str = "default"):
        all_results_data = []
        file_handler = FileHandler()
        completed_urls_list: List[Dict] = []
        completed_urls_lookup: Dict[str, Dict] = {}
        failed_urls_data: Dict[str, Dict] = {}
        formatter = OutputFormatter.get_formatter(format_name)

        if os.path.exists(self.cache_filename):
            with open(self.cache_filename, 'r') as f:
                cached_data = json.load(f)
                if 'urls' in cached_data and isinstance(cached_data['urls'], list):
                    completed_urls_list = cached_data['urls']
                    for item in completed_urls_list:
                        completed_urls_lookup[item['url']] = item
            logger.info(f"Loaded {len(completed_urls_list)} completed URLs from cache.")

        if os.path.exists(self.failed_log_filename):
            with open(self.failed_log_filename, 'r') as f:
                failed_urls_list = json.load(f)
                for item in failed_urls_list:
                    failed_urls_data[item['url']] = item
            logger.info(f"Loaded {len(failed_urls_data)} failed URLs from previous runs.")

        urls_to_skip = set(completed_urls_lookup.keys()).union(set(failed_urls_data.keys()))
        urls_to_crawl = [url for url in urls if url not in urls_to_skip]
        logger.info(f"Found {len(urls_to_crawl)} new URLs to crawl.")

        with tqdm(total=len(urls_to_crawl), desc="Processing URLs") as pbar:
            for url in urls_to_crawl:
                start_time = time.time()
                logger.info(f"Processing URL: {url}")
                crawler = self.get_crawler(url)
                if crawler:
                    results = await crawler.arun(url=url, save_to_file=False)
                    for result in results:
                        end_time = time.time()
                        duration = round(end_time - start_time, 2)
                        if result.success:
                            logger.info(f"Saving images for {url}")
                            saved_images = await crawler._save_images(result, ".jpg")
                            result.images = saved_images

                            prepared_data = formatter(result)
                            all_results_data.append(prepared_data)

                            content_length = len(str(prepared_data))
                            timestamp = datetime.now().isoformat()
                            new_completed_url_entry = {'url': url, 'length': content_length, 'timestamp': timestamp, 'duration': duration}
                            completed_urls_list.append(new_completed_url_entry)
                            completed_urls_lookup[url] = new_completed_url_entry

                            cache_to_save = {'length': len(completed_urls_list), 'urls': completed_urls_list}
                            with open(self.cache_filename, 'w') as f:
                                json.dump(cache_to_save, f, indent=4)

                        else:
                            logger.error(f"  Failed to crawl {url}: {result.error}")
                            timestamp = datetime.now().isoformat()
                            failed_urls_data[url] = {'url': url, 'reason': result.error, 'timestamp': timestamp, 'duration': duration}
                else:
                    end_time = time.time()
                    duration = round(end_time - start_time, 2)
                    logger.warning(f"--- No crawler found for {url} ---")
                    timestamp = datetime.now().isoformat()
                    failed_urls_data[url] = {'url': url, 'reason': "No crawler found", 'timestamp': timestamp, 'duration': duration}
                pbar.update(1)

        if all_results_data:
            output_path = os.path.join("data", "json", output_filename)
            logger.info(f"\nAppending {len(all_results_data)} new results to {output_path}...")
            existing_data = []
            if os.path.exists(output_path):
                with open(output_path, 'r') as f:
                    try:
                        existing_data = json.load(f)
                    except json.JSONDecodeError:
                        logger.warning(f"Warning: Could not decode JSON from {output_path}. Starting with a new file.")
            
            existing_data.extend(all_results_data)

            file_handler.write(
                format_name="json",
                data=existing_data,
                file_name=output_filename, # Use output_filename directly
            )
            logger.info(f"--- All results saved to {output_path} ---")
        else:
            logger.info("\n--- No new results to save. ---")

        if failed_urls_data:
            with open(self.failed_log_filename, 'w') as f:
                json.dump(list(failed_urls_data.values()), f, indent=4)
            logger.info(f"--- {len(failed_urls_data)} failed URLs saved to {self.failed_log_filename} ---")

        total_crawled = len(completed_urls_list) + len(failed_urls_data)
        logger.info(f"\n--- Crawling Summary ---")
        logger.info(f"Total URLs processed: {total_crawled}")
        logger.info(f"Successfully crawled: {len(completed_urls_list)} URLs")
        logger.info(f"Failed to crawl: {len(failed_urls_data)} URLs")
