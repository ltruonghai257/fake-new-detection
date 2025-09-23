import os
import re
from typing import Optional

from ..exceptions import InvalidExtensionException


class StringHandler:
    @staticmethod
    def is_valid_str(ori_str: str) -> bool:
        """
        Check if the given string is a valid non-empty string.
        Args:
            ori_str (str): The string to check.
        Returns:
            bool: True if the string is valid, False otherwise.
        """
        return isinstance(ori_str, str) and bool(ori_str.strip())

    @staticmethod
    def is_valid_extension(ext: str) -> bool:
        """
        Check if the given string is a valid file extension.
        Args:
            ext (str): The file extension to check.
        Returns:
            bool: True if the string is a valid file extension, False otherwise.
        """
        if not bool(re.match(r"^\.[a-zA-Z0-9]+$", ext)):
            raise InvalidExtensionException(ext=ext)
        return True

    @staticmethod
    def is_url(ori_str: str) -> bool:
        """
        Check if the given string is a URL.
        Args:
            ori_str (str): The string to check.
        Returns:
            bool: True if the string is a URL, False otherwise.
        """
        return bool(re.match(r"^https?://[^\s/$.?#].\S*$", ori_str))

    @staticmethod
    def sanitize_filename(filename: str, ext: Optional[str] = None) -> str:
        """
        Sanitize a string to be used as a valid filename.
        Args:
            filename (str): The original filename string.
            ext (Optional[str]): The file extension, including the dot (e.g., ".txt").
        Returns:
            str: A sanitized filename string.
        Example:
            "my document: version 1.txt" -> "my_document_version_1.txt"
            "XMLHttpRequest.js" -> "XMLHttpRequest.js"
        """
        # Split filename and extension
        base, ext_ = os.path.splitext(filename)
        if not ext:
            ext = ext_
        if StringHandler.is_valid_extension(ext):
            # Remove invalid characters for filenames (extended set)
            sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", base)

            # Replace spaces and dots with underscores, preserving case
            sanitized = re.sub(r"[\s.]+", "_", sanitized)

            # Remove leading/trailing underscores
            sanitized = re.sub(r"^_+|_+$", "", sanitized)

            # Replace multiple consecutive underscores with single one
            sanitized = re.sub(r"_+", "_", sanitized)

            # If name becomes empty after sanitization
            if not sanitized:
                sanitized = ""

            # Recombine with extension and limit length
            final_name = sanitized + ext

            # Ensure total length is within limits (255 chars)
            if len(final_name) > 255:
                ext_len = len(ext)
                return sanitized[: (255 - ext_len)] + ext

            return final_name
        return filename

    @staticmethod
    def class_name_to_snake_case(class_name_str: str) -> str:
        """
        Advanced version that handles more edge cases.
        Convert a CamelCase class name string to snake_case.
        Args:
            class_name_str (str): The CamelCase class name string.
        Returns:
            str: The converted snake_case string.
        Example:
            "MyClassName" -> "my_class_name"
            "XMLHttpRequest" -> "xml_http_request"
            "VNExpressCrawler" -> "vn_express_crawler"
        """
        s1 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", class_name_str)
        s2 = re.sub("([A-Z])([A-Z][a-z])", r"\1_\2", s1)
        return s2.lower()

    @staticmethod
    def truncate_string(input_string: str, max_length: int) -> str:
        """
        Truncate a string to a specified maximum length, adding ellipsis if truncated.
        Args:
            input_string (str): The original string.
            max_length (int): The maximum allowed length of the string.
        Returns:
            str: The truncated string with ellipsis if it was longer than max_length.
        Example:
            ("This is a long string", 10) -> "This is a ..."
            ("Short", 10) -> "Short"
        """
        if len(input_string) <= max_length:
            return input_string
        if max_length <= 3:
            return input_string[:max_length]
        return input_string[: max_length - 3] + "..."

    @staticmethod
    def count_words(text):
        """
        Count the number of words in a given text.
        Args:
            text: str: The input text.
        Returns:
            int: The number of words in the text.
        Examples:
            "Hello, world!" -> 2
            "   Leading and trailing spaces   " -> 4
            "" -> 0
            None -> 0

        """
        if not isinstance(text, str):
            return 0
        text = text.strip()

        if not text:
            return 0

        words = text.split()

        return len(words)
