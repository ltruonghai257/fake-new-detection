import asyncio
import os

from crawler.crawler_factory import CrawlerFactory
from processing.dataset_handler import DatasetHandler


import asyncio
import os

from crawler.crawler_factory import CrawlerFactory
from processing.dataset_handler import DatasetHandler


async def run_all_splits(mode: str):
    """
    Runs the crawling process for all splits with either "yes all" or "no all" mode.

    Args:
        mode: "yes_all" to clear cache and crawl all, "no_all" to crawl without clearing cache.
    """
    dataset_name = "tranthaihoa/vifactcheck"
    url_column = "Url"
    url_limit = 5
    splits = ["dev", "test", "train"]

    for split in splits:
        print(f"\n--- Running {mode.replace('_', ' ').title()} scenario for split: {split} ---")
        cache_filename = f"data/caches/crawling_status_{split}_agent_test.json"
        failed_log_filename = f"data/caches/failed_urls_{split}_agent_test.json"
        output_filename = f"news_data_{dataset_name.split('/')[-1]}_{split}_agent_test.json"

        crawler_factory = CrawlerFactory(
            cache_filename=cache_filename,
            failed_log_filename=failed_log_filename
        )
        dataset_handler = DatasetHandler(dataset_name)
        urls_to_crawl = dataset_handler.get_urls_from_split(
            split=split, url_column=url_column, limit=url_limit
        )

        if not urls_to_crawl:
            print(f"--- No URLs to crawl for split {split}. Skipping. ---")
            continue

        if mode == "yes_all":
            print(f"--- Clearing cache for {split} ---")
            crawler_factory.clear_cache()
            await crawler_factory.crawl_and_save_all(urls_to_crawl, output_filename, format_name="custom")
        elif mode == "no_all":
            print(f"--- Crawling for {split} without clearing cache (expecting cached URLs to be skipped) ---")
            await crawler_factory.crawl_and_save_all(urls_to_crawl, output_filename, format_name="custom")
        else:
            print(f"--- Invalid mode: {mode}. Skipping split {split}. ---")


async def main():
    """
    Main function to orchestrate the test scenarios.
    """
    # First, run "yes_all" to populate caches for all splits
    await run_all_splits("yes_all")

    # Then, run "no_all" to check if all are cached
    await run_all_splits("no_all")


if __name__ == "__main__":
    # Set OPENSSL_CONF environment variable
    os.environ["OPENSSL_CONF"] = "openssl.cnf"
    asyncio.run(main())
