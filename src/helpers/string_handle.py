import os
import re
from typing import Optional

from exceptions import InvalidExtensionException


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
        if not isinstance(ori_str, str):
            return False
        return bool(re.match(r"^https?://[^\s/$.?#].\S*$", ori_str))

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize a string to be used as a valid filename.
        """
        return "".join(c if c.isalnum() else "_" for c in filename)

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
    @staticmethod
    def is_valid_url_path(url_path: str) -> bool:
        """
        Check if the given string is a valid URL path for crawling.
        It allows absolute URLs (http, https), relative paths, and protocol-relative URLs.
        It excludes javascript:, mailto:, tel: links.
        """
        if not isinstance(url_path, str) or not url_path.strip():
            return False

        # Exclude javascript, mailto, tel links
        if url_path.startswith(("javascript:", "mailto:", "tel:")):
            return False
            
        # Allow relative paths, absolute paths, and protocol-relative URLs
        return True
