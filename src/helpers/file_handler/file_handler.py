import os
from typing import Union, TypedDict, Optional

from exceptions import InvalidExtensionException


class FileHandler:
    """Base class for file handling operations."""

    def __init__(self, root_folder: str = "data") -> None:
        self.root_folder = root_folder
        self.supported_extensions = tuple(
            [
                ".txt",
                ".text",
                ".md",
                ".markdown",
                ".json",
                ".csv",
                ".jpg",
                ".jpeg",
                ".png",
                ".gif",
                ".bmp",
                ".svg",
            ]
        )

    def mkdir_if_not_exists(self, directory: str) -> None:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory {directory} created.")

    def generate_file_path(
        self, format_name: str, class_name: Optional[str], file_name: str
    ) -> str:
        if class_name:
            folder_path = os.path.join(self.root_folder, format_name, class_name)
        else:
            folder_path = os.path.join(self.root_folder, format_name)
        self.mkdir_if_not_exists(folder_path)
        return os.path.join(folder_path, file_name)

    def read_file(self, file_path: str) -> dict:
        raise NotImplementedError("Subclasses must implement this method.")

    def write_file(self, file_path: str, data: dict) -> None:
        raise NotImplementedError("Subclasses must implement this method.")

    def is_valid_file(self, file_path: str) -> bool:

        return os.path.isfile(file_path) and file_path.endswith(
            self.supported_extensions
        )

    def read(self, file_path: str) -> Union[str, dict, list, TypedDict]:
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
        ext = file_path.split(".")[-1].lower()
        if ext in ("txt", "text", "md", "markdown"):
            from .txt_file_handler import TxtFileHandler

            return TxtFileHandler().read_file(file_path)
        elif ext == "json":
            from .json_file_handler import JsonFileHandler

            return JsonFileHandler().read_file(file_path=file_path)
        elif ext == "csv":
            from .csv_file_handler import CSVFileHandler

            return CSVFileHandler().read_file(file_path)
        elif ext in ("jpg", "jpeg", "png", "gif", "bmp", "svg"):
            from .image_file_handler import ImageFileHandler

            return ImageFileHandler().read_file(file_path)
        else:
            raise InvalidExtensionException(ext=ext)

    def write(
        self,
        format_name: str,
        data: Union[str, dict, list, TypedDict, bytes],
        class_name: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> None:
        """
        Write the content to a file.
        Args:
            format_name: str - The format of the file (e.g., 'json', 'txt').
            data: Union[str, dict, list | TypedDict] - The content to write to the file.
            class_name: Optional[str] - The name of the class using the file handler.
            file_name: Optional[str] - The name of the file.

        Returns:
            None
        Raises:
            InvalidExtensionException: If the file extension is not supported.
            Exception: If there is an error writing to the file.
        """
        if not file_name:
            raise ValueError("file_name must be provided")

        file_path = self.generate_file_path(format_name, class_name, file_name)
        ext = file_name.split(".")[-1].lower()
        if ext in ("txt", "text", "md", "markdown"):
            from .txt_file_handler import TxtFileHandler

            if isinstance(data, str):
                data = {"content": data}
            TxtFileHandler().write_file(file_path, data)
        elif ext == "json":
            from .json_file_handler import JsonFileHandler

            if not isinstance(data, (dict, list)):
                raise ValueError("Data must be a dictionary or a list for JSON files.")
            JsonFileHandler().write_file(file_path, data)
        elif ext == "csv":
            from .csv_file_handler import CSVFileHandler

            if not isinstance(data, list) or not all(
                isinstance(item, dict) for item in data
            ):
                raise ValueError("Data must be a list of dictionaries for CSV files.")
            CSVFileHandler().write_file(file_path, data=data)
        elif ext in ("jpg", "jpeg", "png", "gif", "bmp", "svg"):
            from .image_file_handler import ImageFileHandler

            if isinstance(data, bytes):
                data = {"content": data}
            ImageFileHandler().write_file(file_path, data)
        else:
            raise InvalidExtensionException(ext=ext)
