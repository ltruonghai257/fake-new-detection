class StringException(Exception):
    """Custom exception for string-related errors."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"StringException: {self.message}"


class URLFormatException(StringException):
    """Exception raised for errors in the URL format."""

    def __init__(self, url: str):
        message = f"The URL '{url}' is not in a valid format."
        super().__init__(message)
        self.url = url


class InvalidExtensionException(StringException):
    """Exception raised for invalid file extensions."""

    def __init__(self, ext: str):
        message = f"The file extension '{ext}' is not valid."
        super().__init__(message)
        self.ext = ext
