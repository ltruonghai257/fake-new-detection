import os
import pytest
from src.helpers.json_helper import extract_fields_from_json

@pytest.fixture
def test_data_path():
    """
    Returns the path to the test JSON data file.
    """
    return os.path.join(os.path.dirname(__file__), "test_data.json")

def test_extract_fields_from_json(test_data_path):
    """
    Tests the extract_fields_from_json function.
    """
    # Test extracting 'title' and 'author'
    fields_to_extract = ["title", "author"]
    expected_data = [
        {"title": "Test Title 1", "author": "Author 1"},
        {"title": "Test Title 2", "author": "Author 2"},
        {"title": "Test Title 3"},
    ]
    extracted_data = extract_fields_from_json(test_data_path, fields_to_extract)
    assert extracted_data == expected_data

    # Test extracting only 'id'
    fields_to_extract = ["id"]
    expected_data = [
        {"id": 1},
        {"id": 2},
        {"id": 3},
    ]
    extracted_data = extract_fields_from_json(test_data_path, fields_to_extract)
    assert extracted_data == expected_data

    # Test extracting a field that doesn't exist in all items
    fields_to_extract = ["content", "author"]
    expected_data = [
        {"content": "This is the content of the first item.", "author": "Author 1"},
        {"content": "This is the content of the second item.", "author": "Author 2"},
        {"content": "This is the content of the third item."},
    ]
    extracted_data = extract_fields_from_json(test_data_path, fields_to_extract)
    assert extracted_data == expected_data

    # Test extracting no fields
    fields_to_extract = []
    expected_data = [{}, {}, {}]
    extracted_data = extract_fields_from_json(test_data_path, fields_to_extract)
    assert extracted_data == expected_data

    # Test extracting a field that does not exist in the data
    fields_to_extract = ["non_existent_field"]
    expected_data = []
    extracted_data = extract_fields_from_json(test_data_path, fields_to_extract)
    assert extracted_data == expected_data
