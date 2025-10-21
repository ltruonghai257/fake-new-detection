from .file_handler import FileHandler

class ImageFileHandler(FileHandler):
    def read_file(self, file_path: str) -> dict:
        with open(file_path, "rb") as f:
            content = f.read()
            return {"content": content}

    def write_file(self, file_path: str, data: dict) -> None:
        content = data.get("content")
        if isinstance(content, bytes):
            with open(file_path, "wb") as f:
                f.write(content)
        else:
            raise TypeError("Image content must be in bytes.")
