import json
from typing import Union, List

from .file_handler import FileHandler


class JsonFileHandler(FileHandler):
    """Handles reading and writing JSON files."""

    def __init__(self):
        super().__init__()
        self.supported_extensions = (".json",)

    def read_file(self, file_path: str) -> Union[dict, List]:
        if not self.is_valid_file(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def write_file(self, file_path: str, data: Union[dict, List]) -> None:

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
