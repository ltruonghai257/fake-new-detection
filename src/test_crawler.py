import asyncio
import os
from typing import List

from crawler.crawler_factory import CrawlerFactory
from processing.dataset_handler import DatasetHandler


async def main():
    """
    Main function to load URLs from the dataset and start the crawling process.
    """
    dataset_name = "tranthaihoa/vifactcheck"
    url_column = "Url"
    splits = ["train", "test", "dev"]
    for split in splits:
        output_filename = f"news_data_{dataset_name.split('/')[-1]}_{split}.json"
        # Set a limit for testing. Set to None to crawl all URLs.
        url_limit = 15

        crawler_factory = CrawlerFactory(cache_filename=f"data/caches/crawling_status_{split}.json")

        if crawler_factory.check_cache_file_exists():
            clear_cache_input = input(f"Cache file for split '{split}' exists. Do you want to clear it? (y/n): ")
        
            if clear_cache_input.lower() == "y":
                crawler_factory.clear_cache()
        else:
            print(f"No cache file found for split '{split}'. Starting fresh crawl.")

        dataset_handler = DatasetHandler(dataset_name)
        urls_to_crawl = dataset_handler.get_urls_from_split(
            split=split, url_column=url_column, limit=url_limit
        )

        if urls_to_crawl:
            await crawler_factory.crawl_and_save_all(urls_to_crawl, output_filename, format_name="custom")
        else:
            print("--- No URLs to crawl. Exiting. ---")


if __name__ == "__main__":
    # Set OPENSSL_CONF environment variable
    os.environ["OPENSSL_CONF"] = "openssl.cnf"
    asyncio.run(main())
