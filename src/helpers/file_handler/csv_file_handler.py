import csv

from .file_handler import FileHandler


class CSVFileHandler(FileHandler):
    """Handles reading and writing CSV files."""

    def __init__(self):
        super().__init__()
        self.supported_extensions = (".csv",)

    def read_file(self, file_path: str) -> list[dict]:

        if not self.is_valid_file(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)

    def write_file(self, file_path: str, data: list[dict]) -> None:

        if not data:
            raise ValueError("Data is empty. Cannot write to CSV file.")

        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
