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

class CrawlJournal:
    """
    Tracks completed and failed URLs for resumable crawling.

    Responsibilities:
    - Load/save the completed-URL cache (``cache_path``)
    - Load/save the failed-URL log (``failed_path``)
    - Decide which URLs to skip (completed always; failed only when ``retry_failed=False``)
    """

    def __init__(self, cache_path: str, failed_path: str):
        self.cache_path = cache_path
        self.failed_path = failed_path

    def load(self):
        """
        Read both persisted files.

        Returns:
            (completed_list, completed_lookup, prev_failed)
        """
        completed_list: List[Dict] = []
        completed_lookup: Dict[str, Dict] = {}
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r') as f:
                data = json.load(f)
                if 'urls' in data and isinstance(data['urls'], list):
                    completed_list = data['urls']
                    for item in completed_list:
                        completed_lookup[item['url']] = item
            logger.info(f"Loaded {len(completed_list)} completed URLs from cache (will skip).")

        prev_failed: Dict[str, Dict] = {}
        if os.path.exists(self.failed_path):
            with open(self.failed_path, 'r') as f:
                for item in json.load(f):
                    prev_failed[item['url']] = item

        return completed_list, completed_lookup, prev_failed

    def save_checkpoint(self, completed_list: List[Dict]) -> None:
        """Persist the completed-URL cache to disk."""
        if completed_list:
            with open(self.cache_path, 'w') as f:
                json.dump({'length': len(completed_list), 'urls': completed_list}, f)

    def save_failed(self, all_failed: Dict[str, Dict]) -> None:
        """Persist the failed-URL log to disk."""
        if all_failed:
            with open(self.failed_path, 'w') as f:
                json.dump(list(all_failed.values()), f, indent=2)


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
        self._crawler_cache: Dict[str, BaseCrawler] = {}  # Reuse crawlers per domain

    def get_crawler(self, url: str) -> Optional[BaseCrawler]:
        """
        Factory method to get the appropriate crawler for a given URL.
        Caches crawler instances per domain to reuse HTTP connections.
        """
        try:
            domain = urlparse(url).netloc
            for key, crawler_class in self.CRAWLER_MAPPING.items():
                if key in domain:
                    if key not in self._crawler_cache:
                        self._crawler_cache[key] = crawler_class()
                    return self._crawler_cache[key]
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

    async def crawl_and_save_all(
        self,
        urls: List[str],
        output_filename: str,
        format_name: str = "default",
        max_concurrent: int = 15,
        retry_failed: bool = False,
        save_interval: int = 50,
        output_dir: Optional[str] = None,
    ):
        """
        Crawl URLs concurrently with smart caching.

        Behavior:
        - Normal run: skips completed URLs, skips failed URLs
        - retry_failed=True: skips completed, RETRIES previously failed URLs
        - Saves cache every `save_interval` URLs (resumable on crash)

        Args:
            urls: List of URLs to crawl
            output_filename: Output JSON filename
            format_name: Output format
            max_concurrent: Max concurrent requests
            retry_failed: If True, retry previously failed URLs
            save_interval: Save cache checkpoint every N completed URLs
        """
        import asyncio

        journal = CrawlJournal(self.cache_filename, self.failed_log_filename)
        completed_urls_list, completed_urls_lookup, prev_failed = journal.load()

        all_results_data = []
        file_handler = FileHandler()
        failed_urls_data: Dict[str, Dict] = {}
        formatter = OutputFormatter.get_formatter(format_name)
        lock = asyncio.Lock()
        completed_since_save = 0

        # Decide what to crawl
        urls_to_skip = set(completed_urls_lookup.keys())
        if not retry_failed:
            urls_to_skip |= set(prev_failed.keys())
            logger.info(f"Skipping {len(prev_failed)} previously failed URLs (use retry_failed=True to retry).")
        else:
            logger.info(f"Retrying {len(prev_failed)} previously failed URLs.")

        urls_to_crawl = [url for url in urls if url not in urls_to_skip]
        logger.info(f"URLs to crawl: {len(urls_to_crawl)} (skipped: {len(urls) - len(urls_to_crawl)}, concurrency={max_concurrent})")

        if not urls_to_crawl:
            logger.info("Nothing to crawl.")
            self._print_summary(completed_urls_lookup, failed_urls_data, prev_failed)
            return

        sem = asyncio.Semaphore(max_concurrent)

        async def _save_checkpoint():
            """Save cache to disk (for crash recovery)."""
            journal.save_checkpoint(completed_urls_list)

        async def process_url(url, pbar):
            nonlocal completed_since_save
            async with sem:
                start_time = time.time()
                crawler = self.get_crawler(url)
                if crawler:
                    try:
                        results = await crawler.arun(url=url, save_to_file=False)
                        for result in results:
                            duration = round(time.time() - start_time, 2)
                            if result.success:
                                saved_images = await crawler._save_images(result, ".jpg")
                                result.images = saved_images
                                prepared_data = formatter(result)

                                async with lock:
                                    all_results_data.append(prepared_data)
                                    ts = datetime.now().isoformat()
                                    entry = {'url': url, 'length': len(str(prepared_data)), 'timestamp': ts, 'duration': duration}
                                    completed_urls_list.append(entry)
                                    completed_urls_lookup[url] = entry
                                    # Remove from failed if it was a retry
                                    failed_urls_data.pop(url, None)
                                    completed_since_save += 1
                                    if completed_since_save >= save_interval:
                                        await _save_checkpoint()
                                        completed_since_save = 0
                            else:
                                reason = result.error or "Unknown error"
                                async with lock:
                                    failed_urls_data[url] = {'url': url, 'reason': reason[:200], 'timestamp': datetime.now().isoformat(), 'duration': duration}
                    except Exception as e:
                        duration = round(time.time() - start_time, 2)
                        async with lock:
                            failed_urls_data[url] = {'url': url, 'reason': str(e)[:200], 'timestamp': datetime.now().isoformat(), 'duration': duration}
                else:
                    duration = round(time.time() - start_time, 2)
                    async with lock:
                        failed_urls_data[url] = {'url': url, 'reason': "No crawler found", 'timestamp': datetime.now().isoformat(), 'duration': duration}
                pbar.update(1)

        with tqdm(total=len(urls_to_crawl), desc="Crawling") as pbar:
            tasks = [process_url(url, pbar) for url in urls_to_crawl]
            await asyncio.gather(*tasks)

        # Final save: cache
        await _save_checkpoint()

        # Final save: results JSON (append to existing)
        if all_results_data:
            if output_dir:
                output_path = os.path.join(output_dir, output_filename)
            else:
                output_path = os.path.join("data", "json", output_filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            existing_data = []
            if os.path.exists(output_path):
                with open(output_path, 'r') as f:
                    try:
                        existing_data = json.load(f)
                    except json.JSONDecodeError:
                        pass
            existing_data.extend(all_results_data)
            file_handler.write(format_name="json", data=existing_data, file_name=output_filename)
            logger.info(f"Saved {len(all_results_data)} new + {len(existing_data) - len(all_results_data)} existing = {len(existing_data)} total articles to {output_path}")

        # Merge new failures with previous (keep both)
        all_failed = {**prev_failed, **failed_urls_data}
        # Remove any that succeeded on retry
        for url in completed_urls_lookup:
            all_failed.pop(url, None)
        journal.save_failed(all_failed)

        self._print_summary(completed_urls_lookup, failed_urls_data, all_failed)

    def _print_summary(self, completed, new_failed, total_failed):
        logger.info(f"\n--- Crawling Summary ---")
        logger.info(f"Completed (total): {len(completed)} URLs")
        logger.info(f"Failed this run:   {len(new_failed)} URLs")
        logger.info(f"Failed (total):    {len(total_failed)} URLs")
        if total_failed:
            reasons = {}
            for item in total_failed.values():
                r = item.get('reason', 'unknown')[:50]
                reasons[r] = reasons.get(r, 0) + 1
            logger.info(f"Failure reasons:")
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1])[:5]:
                logger.info(f"  {count:4d}x {reason}")
