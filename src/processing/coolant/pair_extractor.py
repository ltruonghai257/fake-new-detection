"""
Extract (image, caption) pairs from crawled ViFactCheck JSON data.

COOLANT uses image-caption pairs where:
- matched pair (caption_i, image_i) = Real
- unmatched pair (caption_i, image_j where i!=j) = Fake

This module extracts valid pairs from crawled articles.
"""

import json
import os
import re
import html as html_lib
from pathlib import Path
from collections import Counter
from typing import List, Dict, Optional


def clean_caption(text: str) -> str:
    """
    Clean a caption string extracted from HTML.

    - Decode HTML entities
    - Remove HTML tags
    - Remove photo credit/attribution lines
    - Remove photo index markers
    - Normalize whitespace
    """
    if not text:
        return ""

    # Decode HTML entities (&amp; -> &, &quot; -> ", etc.)
    text = html_lib.unescape(text)

    # Remove any remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove photo credit/attribution patterns (Vietnamese)
    # "Ảnh: NVCC", "Ảnh: Quang Phúc", "Ảnh minh họa: Reuters", etc.
    text = re.sub(r"\(?[Ảả]nh\s*(minh\s*họa)?\s*[:\.]\s*[^,\.\)]{0,50}\)?\s*$", "", text, flags=re.IGNORECASE).strip()
    # "Nguồn: VnExpress", "Nguồn ảnh: Internet"
    text = re.sub(r"\(?[Nn]guồn\s*(ảnh)?\s*[:\.]\s*[^,\.\)]{0,50}\)?\s*$", "", text, flags=re.IGNORECASE).strip()
    # "Ảnh by ...", "Photo: ...", "Photo by ..."
    text = re.sub(r"\(?([Pp]hoto|[Ảả]nh)\s*(by|bởi)\s*[:\.]*\s*[^,\.\)]{0,50}\)?\s*$", "", text, flags=re.IGNORECASE).strip()
    # "Đồ họa: ...", "Infographic: ..."
    text = re.sub(r"\(?([Đđ]ồ\s*họa|[Ii]nfographic)\s*[:\.]\s*[^,\.\)]{0,50}\)?\s*$", "", text, flags=re.IGNORECASE).strip()
    # "(ảnh: ABC)", "(Nguồn: XYZ)" — parenthesized credits anywhere
    text = re.sub(r"\(\s*(?:[Ảả]nh|[Nn]guồn|[Pp]hoto)\s*[:\.]\s*[^)]{0,50}\)", "", text, flags=re.IGNORECASE).strip()

    # Remove trailing photo index markers (e.g., "- Ảnh 1.", "- Ảnh 2.")
    text = re.sub(r"\s*-\s*[Ảả]nh\s*\d+\.?\s*$", "", text).strip()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove trailing punctuation artifacts
    text = re.sub(r"[\s\-–—:\.]+$", "", text).strip()

    return text


def is_credit_only(text: str) -> bool:
    """Check if cleaned text is ONLY a photo credit with no real caption."""
    if not text:
        return True
    # Pure credit patterns (entire string is just credit)
    credit_patterns = [
        r"^[Ảả]nh\s*(minh\s*họa)?\s*$",          # "Ảnh", "Ảnh minh họa"
        r"^[Ảả]nh\s*\d+\s*$",                      # "Ảnh 1", "Ảnh 2"
        r"^[Nn]guồn\s*(ảnh)?\s*$",                  # "Nguồn", "Nguồn ảnh"
        r"^[Pp]hoto\s*$",                            # "Photo"
        r"^NVCC$",                                    # "NVCC" (nguồn viết cung cấp)
        r"^[A-Z]{2,6}$",                             # Pure abbreviations like "AFP", "TTXVN"
        r"^Không được đề cập$",                      # Default placeholder
    ]
    for pattern in credit_patterns:
        if re.match(pattern, text.strip()):
            return True
    return False


class PairExtractor:
    """Extract (image_path, caption) pairs from crawled JSON articles."""

    DEFAULT_CAPTION = "Không được đề cập"

    def __init__(self, jpg_base_dir: str, min_caption_len: int = 5):
        """
        Args:
            jpg_base_dir: Base directory where JPG images are stored
                (e.g., notebooks/data/jpg)
            min_caption_len: Minimum caption length to consider valid
        """
        self.jpg_base_dir = Path(jpg_base_dir)
        self.min_caption_len = min_caption_len

    def extract_from_json(self, json_path: str) -> List[Dict]:
        """
        Extract valid (image, caption) pairs from a crawled JSON file.

        Args:
            json_path: Path to news_data_vifactcheck_*_cleaned.json

        Returns:
            List of dicts with keys: image_path, caption, article_idx
        """
        with open(json_path, "r", encoding="utf-8") as f:
            articles = json.load(f)

        pairs = []
        skipped = {"no_caption": 0, "credit_only": 0, "too_short": 0, "no_image": 0}
        for article_idx, article in enumerate(articles):
            for img in article.get("images", []):
                raw_caption = (img.get("caption") or "").strip()
                folder_path = img.get("folder_path", "")

                # Must have an image file
                if not folder_path:
                    skipped["no_image"] += 1
                    continue

                # Normalize path separators (Windows backslash -> forward slash)
                folder_path = folder_path.replace("\\", "/")

                # Resolve full path — try multiple base strategies
                # folder_path can be "jpg/source/file.jpg" or "source/file.jpg"
                full_path = self.jpg_base_dir / folder_path
                if not full_path.exists():
                    # Try stripping leading "jpg/" if jpg_base_dir already ends with /jpg
                    if folder_path.startswith("jpg/"):
                        alt_path = self.jpg_base_dir / folder_path[4:]
                        if alt_path.exists():
                            full_path = alt_path
                            folder_path = folder_path[4:]
                if not full_path.exists():
                    # Try parent of jpg_base_dir (e.g., notebooks/data instead of notebooks/data/jpg)
                    alt_path = self.jpg_base_dir.parent / folder_path
                    if alt_path.exists():
                        full_path = alt_path
                if not full_path.exists():
                    skipped["no_image"] += 1
                    continue

                # Clean the caption
                caption = clean_caption(raw_caption)

                # Skip empty / default / placeholder
                if not caption or caption == self.DEFAULT_CAPTION:
                    skipped["no_caption"] += 1
                    continue

                # Skip credit-only captions (e.g., "Ảnh: NVCC", "AFP")
                if is_credit_only(caption):
                    skipped["credit_only"] += 1
                    continue

                # Skip too short
                if len(caption) < self.min_caption_len:
                    skipped["too_short"] += 1
                    continue

                pairs.append({
                    "image_path": str(full_path),
                    "caption": caption,
                    "article_idx": article_idx,
                    "folder_path": folder_path,
                })

        total_imgs = sum(len(a.get("images", [])) for a in articles)
        print(f"    Total images: {total_imgs}, Valid pairs: {len(pairs)}")
        print(f"    Skipped: {skipped}")

        return pairs

    def extract_all_splits(
        self,
        json_dir: str,
        splits: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict]]:
        """
        Extract pairs from all splits (train/dev/test).

        Args:
            json_dir: Directory containing news_data_vifactcheck_*_cleaned.json
            splits: List of split names (default: train, dev, test)

        Returns:
            Dict mapping split name to list of pairs
        """
        if splits is None:
            splits = ["train", "dev", "test"]

        json_dir = Path(json_dir)
        all_pairs = {}

        for split in splits:
            json_path = json_dir / f"news_data_vifactcheck_{split}_cleaned.json"
            if not json_path.exists():
                # Try without _cleaned suffix
                json_path = json_dir / f"news_data_vifactcheck_{split}.json"
            if not json_path.exists():
                print(f"  {split}: file not found, skipping")
                continue

            pairs = self.extract_from_json(str(json_path))
            all_pairs[split] = pairs

        return all_pairs

    @staticmethod
    def print_stats(pairs: List[Dict], split_name: str = ""):
        """Print statistics about extracted pairs."""
        n = len(pairs)
        if n == 0:
            print(f"  {split_name}: 0 pairs")
            return

        caption_lens = [len(p["caption"]) for p in pairs]
        article_ids = set(p["article_idx"] for p in pairs)
        avg_len = sum(caption_lens) / n
        min_len = min(caption_lens)
        max_len = max(caption_lens)

        print(f"  {split_name}: {n} pairs from {len(article_ids)} articles")
        print(f"    Caption length: avg={avg_len:.0f}, min={min_len}, max={max_len}")

    @staticmethod
    def save_pairs(pairs: List[Dict], output_path: str):
        """Save extracted pairs to JSON."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(pairs)} pairs to {output_path}")

    @staticmethod
    def load_pairs(json_path: str) -> List[Dict]:
        """Load pairs from JSON."""
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
