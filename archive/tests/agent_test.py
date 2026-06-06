import asyncio
import os

from crawler.crawler_factory import CrawlerFactory
from processing.dataset_handler import DatasetHandler
from processing.data_converter import DataConverter # Import the new DataConverter class


async def main():
    """
    Main function for the agent to test the crawling process with specific URLs from failed_urls_train.json.
    """
    dataset_name = "tranthaihoa/vifactcheck"
    url_column = "Url"
    split = "test" # Using 'test' split for agent_test output files
    output_json_filename = f"news_data_{dataset_name.split('/')[-1]}_{split}_agent_test.json"
    output_csv_filename = f"news_data_{dataset_name.split('/')[-1]}_{split}_agent_test.csv"
    # url_limit = 5 # Not needed when providing specific URLs

    crawler_factory = CrawlerFactory(
        cache_filename=f"data/caches/crawling_status_{split}_agent_test.json",
        failed_log_filename=f"data/caches/failed_urls_{split}_agent_test.json"
    )

    # Clear cache to ensure fresh crawl for testing
    if crawler_factory.check_cache_file_exists():
        crawler_factory.clear_cache()

    # Clear failed URLs cache for agent_test to re-attempt
    failed_urls_agent_test_path = os.path.join("data", "caches", f"failed_urls_{split}_agent_test.json")
    if os.path.exists(failed_urls_agent_test_path):
        os.remove(failed_urls_agent_test_path)
        print(f"--- Cleared {failed_urls_agent_test_path} ---")

    # URLs from failed_urls_train.json
    urls_to_crawl = [
        "https://baotintuc.vn/doi-song-van-hoa/dong-bao-cham-o-binh-thuan-phan-khoi-don-tet-ramuwan-20230321154228225.htm",
        "https://baotintuc.vn/dich-benh/benh-dai-dien-bien-phuc-tap-tai-dong-nai-20230324164409440.htm",
        "https://baotintuc.vn/giao-duc/cong-bo-quyet-dinh-thanh-lap-phan-hieu-dai-hoc-thai-nguyen-tai-ha-giang-20230313211018162.htm",
        "https://baotintuc.vn/the-gioi/liban-ghi-nhan-sieu-lam-phat-thang-thu-32-lien-tiep-20230401062929962.htm"
    ]

    if urls_to_crawl:
        await crawler_factory.crawl_and_save_all(urls_to_crawl, output_json_filename, format_name="custom")
        
        # Convert the generated JSON to CSV using DataConverter
        json_file_path = os.path.join("data", "json", output_json_filename)
        csv_file_path = os.path.join("data", "csv", output_csv_filename)
        
        os.makedirs(os.path.join("data", "csv"), exist_ok=True)

        data_converter = DataConverter()
        data_converter.convert_json_to_csv(json_file_path, csv_file_path)
    else:
        print("--- No URLs to crawl for agent test. Exiting. ---")


if __name__ == "__main__":
    from helpers.legacy_tool_handler import LegacyToolHandler
    # Set OPENSSL_CONF environment variable using LegacyToolHandler
    with LegacyToolHandler(openssl_conf_path="openssl.cnf"):
        asyncio.run(main())
