import csv
import json
import xml.etree.ElementTree as ET
from io import StringIO
from typing import Dict, List, Union, Any, Optional


class Parser:
    """
    Một class Parser tổng quát để parse nhiều định dạng dữ liệu khác nhau
    """

    def __init__(self, data: str = None, file_path: str = None):
        """
        Khởi tạo Parser

        Args:
            data: Chuỗi dữ liệu cần parse
            file_path: Đường dẫn đến file cần parse
        """
        self.data = data
        self.file_path = file_path

    def parse_json(self) -> Union[Dict, List]:
        """Parse JSON từ string hoặc file"""
        try:
            if self.file_path:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            elif self.data:
                return json.loads(self.data)
            else:
                raise ValueError("Cần cung cấp data hoặc file_path")
        except json.JSONDecodeError as e:
            raise ValueError(f"Lỗi parse JSON: {e}")

    def parse_csv(
        self, delimiter: str = ",", has_header: bool = True
    ) -> List[Optional[Dict, Any]]:
        """
        Parse CSV từ string hoặc file

        Args:
            delimiter: Ký tự phân cách (mặc định là ',')
            has_header: File có header hay không

        Returns:
            List các dictionary (nếu có header) hoặc list các list
        """
        try:
            if self.file_path:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    if has_header:
                        reader = csv.DictReader(f, delimiter=delimiter)
                        return list(reader)
                    else:
                        reader = csv.reader(f, delimiter=delimiter)
                        return list(reader)
            elif self.data:
                if has_header:
                    reader = csv.DictReader(StringIO(self.data), delimiter=delimiter)
                    return list(reader)
                else:
                    reader = csv.reader(StringIO(self.data), delimiter=delimiter)
                    return list(reader)
            else:
                raise ValueError("Cần cung cấp data hoặc file_path")
        except Exception as e:
            raise ValueError(f"Lỗi parse CSV: {e}")

    def parse_xml(self) -> ET.Element:
        """Parse XML từ string hoặc file"""
        try:
            if self.file_path:
                tree = ET.parse(self.file_path)
                return tree.getroot()
            elif self.data:
                return ET.fromstring(self.data)
            else:
                raise ValueError("Cần cung cấp data hoặc file_path")
        except ET.ParseError as e:
            raise ValueError(f"Lỗi parse XML: {e}")

    def xml_to_dict(self, element: ET.Element = None) -> Dict[str, Any]:
        """
        Chuyển XML Element thành dictionary

        Args:
            element: XML Element (nếu None sẽ parse từ data/file_path)
        """
        if element is None:
            element = self.parse_xml()

        result = {element.tag: {} if element.attrib else None}
        children = list(element)

        if children:
            dd = {}
            for dc in map(self.xml_to_dict, children):
                for k, v in dc.items():
                    if k in dd:
                        if isinstance(dd[k], list):
                            dd[k].append(v)
                        else:
                            dd[k] = [dd[k], v]
                    else:
                        dd[k] = v
            result = {element.tag: dd}

        if element.attrib:
            result[element.tag].update(("@" + k, v) for k, v in element.attrib.items())

        if element.text:
            text = element.text.strip()
            if children or element.attrib:
                if text:
                    result[element.tag]["#text"] = text
            else:
                result[element.tag] = text

        return result

    def parse_key_value(
        self, delimiter: str = "=", comment_char: str = "#"
    ) -> Dict[str, str]:
        """
        Parse định dạng key-value (như .env, .ini, .properties)

        Args:
            delimiter: Ký tự phân cách key và value
            comment_char: Ký tự bắt đầu comment
        """
        result = {}
        lines = (
            self.data.split("\n")
            if self.data
            else open(self.file_path, "r").readlines()
        )

        for line in lines:
            line = line.strip()
            if not line or line.startswith(comment_char):
                continue

            if delimiter in line:
                key, value = line.split(delimiter, 1)
                result[key.strip()] = value.strip()

        return result

    def parse_html(self) -> str:
        """
        Parse HTML từ string hoặc file

        Returns:
            Chuỗi HTML
        """
        try:
            if self.file_path:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    return f.read()
            elif self.data:
                return self.data
            else:
                raise ValueError("Cần cung cấp data hoặc file_path")
        except Exception as e:
            raise ValueError(f"Lỗi parse HTML: {e}")

    @staticmethod
    def auto_detect_format(data: str) -> str:
        """
        Tự động phát hiện định dạng dữ liệu

        Returns:
            'json', 'xml', 'csv', hoặc 'unknown'
        """
        data = data.strip()

        if data.startswith("{") or data.startswith("["):
            return "json"
        elif data.startswith("<"):
            return "xml"
        elif "," in data or "\t" in data:
            return "csv"
        else:
            return "unknown"
