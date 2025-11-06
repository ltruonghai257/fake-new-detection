import json
import csv
import os
from typing import List, Dict, Any

class DataConverter:
    def convert_json_to_csv(self, json_file_path: str, csv_file_path: str):
        """
        Converts a JSON file (list of dictionaries) to a CSV file, flattening nested structures.
        """
        if not os.path.exists(json_file_path):
            print(f"Error: JSON file not found at {json_file_path}")
            return

        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            print("Error: JSON data is not a list of dictionaries. Cannot convert to CSV.")
            return

        if not data:
            print("Warning: JSON file is empty. No CSV will be generated.")
            return

        # Determine all unique headers (keys) from the JSON data
        fieldnames = set()
        for item in data:
            for key, value in item.items():
                if isinstance(value, dict):
                    for sub_key in value.keys():
                        fieldnames.add(f"{key}_{sub_key}")
                else:
                    fieldnames.add(key)
        fieldnames = sorted(list(fieldnames)) # Sort for consistent column order

        with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for item in data:
                row = {}
                for key, value in item.items():
                    if isinstance(value, list):
                        # Join list items with a separator
                        if key == "images":
                            # For images, convert each dict to JSON string
                            row[key] = "|".join([json.dumps(img) for img in value])
                        elif key == "other_urls":
                            row[key] = "|".join(value)
                        elif key == "contents":
                            row[key] = "\n\n".join(value) # Use double newline for paragraphs
                        else:
                            row[key] = "|".join(map(str, value))
                    elif isinstance(value, dict):
                        # Flatten nested dictionaries
                        for sub_key, sub_value in value.items():
                            row[f"{key}_{sub_key}"] = sub_value
                    else:
                        row[key] = value
                writer.writerow(row)
        print(f"Successfully converted {json_file_path} to {csv_file_path}")