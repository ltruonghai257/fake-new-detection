from .file_handler import FileHandler


class TxtFileHandler(FileHandler):
    """
    Handles reading and writing TXT files.
    """

    def read_file(self, file_path: str) -> dict:
        if not self.is_valid_file(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            return {"content": content}

    def write_file(self, file_path: str, data: dict) -> None:
        content = data.get("content", "")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    def is_valid_file(self, file_path: str) -> bool:
        import os

        return os.path.isfile(file_path) and file_path.endswith(".txt")
