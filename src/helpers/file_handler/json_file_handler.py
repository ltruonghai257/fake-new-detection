import json
import os

from .file_handler import FileHandler


class JsonFileHandler(FileHandler):
    """Handles reading and writing JSON files."""

    def read_file(self, file_path: str) -> dict:
        if not self.is_valid_file(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def write_file(self, file_path: str, data: dict) -> None:

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def is_valid_file(self, file_path: str) -> bool:

        return os.path.isfile(file_path) and file_path.endswith(".json")
