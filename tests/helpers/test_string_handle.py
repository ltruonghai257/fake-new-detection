import pytest

from src.helpers import StringHandler


class TestStringHandle:
    @pytest.mark.parametrize(
        "input_string, expected",
        [
            ("http://example.com", True),
            ("https://example.com", True),
            ("ftp://example.com", False),
            ("example.com", False),
            ("http:/example.com", False),
            ("", False),
            ("Just some text", False),
            ("https://sub.domain.com/path?query=string#fragment", True),
        ],
    )
    def test_is_url(self, input_string, expected):

        assert StringHandler.is_url(input_string) == expected

    @pytest.mark.parametrize(
        "input_string, expected",
        [
            ("my document: version 1.txt", "my_document_version_1.txt"),
            ("invalid/filename\\test?.txt", "invalidfilenametest.txt"),
            ("   leading and trailing spaces   ", "leading_and_trailing_spaces"),
            ("a" * 300 + ".txt", "a" * 251 + ".txt"),  # Limit to 255 characters
            ("normal_filename.txt", "normal_filename.txt"),
            ("file*name|with<invalid>chars?.txt", "filenamewithinvalidchars.txt"),
            ("multiple   spaces.txt", "multiple_spaces.txt"),
            ("", ""),
        ],
    )
    def test_sanitize_filename(self, input_string, expected):
        assert StringHandler.sanitize_filename(input_string) == expected

    @pytest.mark.parametrize(
        "input_string, expected",
        [
            ("MyClassName", "my_class_name"),
            ("AnotherExample", "another_example"),
            ("SimpleTest", "simple_test"),
            ("", ""),
            ("Single", "single"),
            ("Already_Snake_Case", "already_snake_case"),
            ("VNExpressCrawler", "vn_express_crawler"),
        ],
    )
    def test_class_name_to_snake_case(self, input_string, expected):
        assert StringHandler.class_name_to_snake_case(input_string) == expected

    @pytest.mark.parametrize(
        "input_string, expected",
        [
            ("Hello, world!", 2),
            ("This is a test string.", 5),
            ("", 0),
            ("OneWord", 1),
            ("Multiple    spaces here", 3),
            ("Newlines\nand\ttabs", 3),
        ],
    )
    def test_count_words(self, input_string, expected):
        assert StringHandler.count_words(input_string) == expected
