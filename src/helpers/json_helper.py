"""
This module provides helper functions for working with JSON files.
"""
import orjson
from typing import List, Any, Dict, Union

def extract_fields_from_json(file_path: str, fields: List[str]) -> List[Dict[str, Any]]:
    """
    Extracts specified fields from a JSON file.

    Args:
        file_path (str): The path to the JSON file.
        fields (List[str]): A list of fields to extract.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                               contains the extracted fields for an item.
    """
    with open(file_path, "rb") as f:
        data = orjson.loads(f.read())

    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of objects.")

    if not fields:
        return [{} for _ in data]

    extracted_data = []
    for item in data:
        extracted_item = {}
        for field in fields:
            if field in item:
                extracted_item[field] = item[field]
        if extracted_item:
            extracted_data.append(extracted_item)

    return extracted_data
