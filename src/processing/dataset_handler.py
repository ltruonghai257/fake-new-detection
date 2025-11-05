from typing import List

class DatasetHandler:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def get_urls_from_split(self, split: str, url_column: str, limit: int = None) -> List[str]:
        """
        Loads a dataset from Hugging Face and extracts URLs from a specified column.
        """
        try:
            from datasets import load_dataset
        except ImportError:
            print("Error: The 'datasets' library is not installed.")
            print("Please run 'pip install datasets huggingface_hub' to install it.")
            return []

        print(f"--- Loading dataset '{self.dataset_name}' from Hugging Face... ---")
        try:
            # Use streaming to avoid downloading the entire dataset at once
            ds = load_dataset(self.dataset_name, split=split, streaming=True)
            urls = []
            dataset_iterator = ds.take(limit) if limit else ds

            print(f"--- Extracting URLs from the '{url_column}' column... ---")
            for item in dataset_iterator:
                if item and item.get(url_column):
                    urls.append(item[url_column])

            print(f"--- Found {len(urls)} URLs to crawl. ---")
            return list(set(urls))  # Return unique URLs
        except Exception as e:
            print(f"Error loading or processing dataset: {e}")
            print("Please ensure you are logged in to Hugging Face.")
            print("You can do this by running 'huggingface-cli login' in your terminal.")
            return []
