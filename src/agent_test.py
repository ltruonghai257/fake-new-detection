import asyncio
import os

from crawler.crawler_factory import CrawlerFactory
from processing.dataset_handler import DatasetHandler


async def main():
    """
    Main function for the agent to test the crawling process.
    """
    dataset_name = "tranthaihoa/vifactcheck"
    url_column = "Url"
    split = "test"
    output_filename = f"news_data_{dataset_name.split('/')[-1]}_{split}_agent_test.json"
    url_limit = 5

    crawler_factory = CrawlerFactory(cache_filename=f"data/caches/crawling_status_{split}_agent_test.json")

    if crawler_factory.check_cache_file_exists():
        crawler_factory.clear_cache()

    dataset_handler = DatasetHandler(dataset_name)
    urls_to_crawl = dataset_handler.get_urls_from_split(
        split=split, url_column=url_column, limit=url_limit
    )

    if urls_to_crawl:
        await crawler_factory.crawl_and_save_all(urls_to_crawl, output_filename, format_name="custom")
    else:
        print("--- No URLs to crawl for agent test. Exiting. ---")


if __name__ == "__main__":
    # Set OPENSSL_CONF environment variable
    os.environ["OPENSSL_CONF"] = "openssl.cnf"
    asyncio.run(main())
