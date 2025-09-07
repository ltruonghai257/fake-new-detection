from typing import Union, TypedDict, Optional

from exceptions import InvalidExtensionException


class FileHandler:
    """Base class for file handling operations."""

    def __init__(self, file_path: str = "") -> None:
        self.file_path = file_path

    def read_file(self, file_path: str) -> dict:
        raise NotImplementedError("Subclasses must implement this method.")

    def write_file(self, file_path: str, data: dict) -> None:
        raise NotImplementedError("Subclasses must implement this method.")

    def is_valid_file(self, file_path: str) -> bool:
        import os

        return os.path.isfile(file_path)

    def read(self, file_path: Optional[str]) -> Union[str, dict, list | TypedDict]:
        """
          Read the content from a file.
        Args:
            file_path: str - The path to the file.
         Raises:
               InvalidExtensionException: If the file extension is not supported.
               Exception: If there is an error reading the file.
        Returns:
               Union[str, dict, list | TypedDict] - The content of the file.
        """
        if not file_path:
            file_path = self.file_path
        ext = file_path.split(".")[-1].lower()
        if ext == "txt":
            from .txt_file_handler import TxtFileHandler

            return TxtFileHandler.read_file(file_path)
        elif ext == "json":
            from .json_file_handler import JsonFileHandler

            return JsonFileHandler.read_file(file_path)
        elif ext == "csv":
            from .csv_file_handler import CSVFileHandler

            return CSVFileHandler.read_file(file_path)
        else:
            raise InvalidExtensionException(ext=ext)

    def write(self, file_path: Optional[str], data: Union[str, dict, list | TypedDict]):
        """
        Write the content to a file.
        Args:
            file_path: str - The path to the file.
            data: Union[str, dict, list | TypedDict] - The content to write to the file.

        Returns:
            None
        Raises:
            InvalidExtensionException: If the file extension is not supported.
            Exception: If there is an error writing to the file.
        Example:
            file_handler = FileHandler("example.txt")
            file_handler.write("example.txt", "Hello, World!")

        """
        if not file_path:
            file_path = self.file_path
        ext = file_path.split(".")[-1].lower()
        if ext == "txt":
            from .txt_file_handler import TxtFileHandler

            if isinstance(data, str):
                data = {"content": data}
            TxtFileHandler.write_file(file_path, data)
        elif ext == "json":
            from .json_file_handler import JsonFileHandler

            if not isinstance(data, dict):
                raise ValueError("Data must be a dictionary for JSON files.")
            JsonFileHandler.write_file(file_path, data)
        elif ext == "csv":
            from .csv_file_handler import CSVFileHandler

            if not isinstance(data, list) or not all(
                isinstance(item, dict) for item in data
            ):
                raise ValueError("Data must be a list of dictionaries for CSV files.")
            CSVFileHandler.write_file(file_path, data=data)
        else:
            raise InvalidExtensionException(ext=ext)
