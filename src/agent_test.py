import asyncio
import os

from crawler.crawler_factory import CrawlerFactory
from processing.dataset_handler import DatasetHandler
from processing.data_converter import DataConverter # Import the new DataConverter class


async def main():
    """
    Main function for the agent to test the crawling process and JSON to CSV conversion.
    """
    dataset_name = "tranthaihoa/vifactcheck"
    url_column = "Url"
    split = "test"
    output_json_filename = f"news_data_{dataset_name.split('/')[-1]}_{split}_agent_test.json"
    output_csv_filename = f"news_data_{dataset_name.split('/')[-1]}_{split}_agent_test.csv"
    url_limit = 5

    crawler_factory = CrawlerFactory(
        cache_filename=f"data/caches/crawling_status_{split}_agent_test.json",
        failed_log_filename=f"data/caches/failed_urls_{split}_agent_test.json"
    )

    # Clear cache to ensure fresh crawl for testing conversion
    if crawler_factory.check_cache_file_exists():
        crawler_factory.clear_cache()

    dataset_handler = DatasetHandler(dataset_name)
    urls_to_crawl = dataset_handler.get_urls_from_split(
        split=split, url_column=url_column, limit=url_limit
    )

    if urls_to_crawl:
        await crawler_factory.crawl_and_save_all(urls_to_crawl, output_json_filename, format_name="custom")
        
        # Convert the generated JSON to CSV using DataConverter
        json_file_path = os.path.join("data", "json", output_json_filename)
        csv_file_path = os.path.join("data", "csv", output_csv_filename) # Save CSV to a 'csv' subdirectory
        
        # Ensure the 'data/csv' directory exists
        os.makedirs(os.path.join("data", "csv"), exist_ok=True)

        data_converter = DataConverter() # Instantiate DataConverter
        data_converter.convert_json_to_csv(json_file_path, csv_file_path)
    else:
        print("--- No URLs to crawl for agent test. Exiting. ---")


if __name__ == "__main__":
    # Set OPENSSL_CONF environment variable
    os.environ["OPENSSL_CONF"] = "openssl.cnf"
    asyncio.run(main())
