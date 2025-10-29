import asyncio
import sys
import os
from typing import List

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.crawler.crawler_factory import crawl_and_save_all

def get_urls_from_dataset(dataset_name: str, split: str, url_column: str, limit: int = None) -> List[str]:
    """
    Loads a dataset from Hugging Face and extracts URLs from a specified column.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: The 'datasets' library is not installed.")
        print("Please run 'pip install datasets huggingface_hub' to install it.")
        return []

    print(f"--- Loading dataset '{dataset_name}' from Hugging Face... ---")
    try:
        # Use streaming to avoid downloading the entire dataset at once
        ds = load_dataset(dataset_name, split=split, streaming=True)
        urls = []
        dataset_iterator = ds.take(limit) if limit else ds
        
        print(f"--- Extracting URLs from the '{url_column}' column... ---")
        for item in dataset_iterator:
            if item and item.get(url_column):
                urls.append(item[url_column])
        
        print(f"--- Found {len(urls)} URLs to crawl. ---")
        return list(set(urls)) # Return unique URLs
    except Exception as e:
        print(f"Error loading or processing dataset: {e}")
        print("Please ensure you are logged in to Hugging Face.")
        print("You can do this by running 'huggingface-cli login' in your terminal.")
        return []

async def main():
    """
    Main function to load URLs from the dataset and start the crawling process.
    """
    dataset_name = "tranthaihoa/vifactcheck"
    split = "train"
    url_column = "evidence_url"
    output_filename = "vifactcheck_crawl_results.json"
    # Set a limit for testing. Set to None to crawl all URLs.
    url_limit = 25

    urls_to_crawl = get_urls_from_dataset(
        dataset_name=dataset_name,
        split=split,
        url_column=url_column,
        limit=url_limit
    )

    if urls_to_crawl:
        await crawl_and_save_all(urls_to_crawl, output_filename)
    else:
        print("--- No URLs to crawl. Exiting. ---")

if __name__ == "__main__":
    asyncio.run(main())
