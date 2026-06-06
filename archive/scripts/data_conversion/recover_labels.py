#!/usr/bin/env python3
"""
Recover ViFactCheck labels from HuggingFace dataset and inject into crawled JSON files.

ViFactCheck labels:
  0 = Supported (Real)
  1 = Refuted (Fake)
  2 = NEI (Not Enough Information)
  None = Missing

Label mapping strategies:
  binary_exclude_nei: Supported→0, Refuted→1, exclude NEI/None
  binary_nei_as_real: Supported→0, NEI→0, Refuted→1, exclude None
  three_class:        Supported→0, Refuted→1, NEI→2, exclude None

Usage:
  python recover_labels.py --strategy binary_exclude_nei
  python recover_labels.py --strategy three_class --dry-run
"""

import json
import argparse
from pathlib import Path
from collections import Counter
from urllib.parse import urlparse, urlunparse


def normalize_url(url: str) -> str:
    """Normalize URL for reliable matching (strip trailing slash, lowercase scheme/host)."""
    if not url:
        return ""
    try:
        parsed = urlparse(url.strip())
        # Lowercase scheme and netloc, strip trailing slash from path
        path = parsed.path.rstrip("/") if parsed.path != "/" else "/"
        normalized = urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            path,
            parsed.params,
            parsed.query,
            ""  # drop fragment
        ))
        return normalized
    except Exception:
        return url.strip().rstrip("/")


def load_hf_labels(split: str, dataset_name: str = "tranthaihoa/vifactcheck") -> dict:
    """
    Load ViFactCheck dataset from HuggingFace and build url→label mapping.
    
    When multiple claims share the same URL, use the most severe label:
    Refuted (1) > NEI (2) > Supported (0).
    
    Returns:
        dict: {normalized_url: label_value}
    """
    from datasets import load_dataset

    print(f"  Loading HF dataset split='{split}'...")
    ds = load_dataset(dataset_name, split=split)

    url_labels = {}  # normalized_url → list of labels
    skipped_none = 0

    for item in ds:
        url = item.get("Url", "")
        label = item.get("labels")

        if not url:
            continue
        if label is None:
            skipped_none += 1
            continue

        norm_url = normalize_url(url)
        if norm_url not in url_labels:
            url_labels[norm_url] = []
        url_labels[norm_url].append(int(label))

    # Resolve conflicts: most severe label wins
    # Severity order: Refuted(1) > NEI(2) > Supported(0)
    severity = {1: 3, 2: 2, 0: 1}
    resolved = {}
    conflicts = 0
    for url, labels in url_labels.items():
        unique = set(labels)
        if len(unique) > 1:
            conflicts += 1
        resolved[url] = max(unique, key=lambda x: severity.get(x, 0))

    print(f"  HF {split}: {len(ds)} entries → {len(resolved)} unique URLs "
          f"(skipped {skipped_none} None labels, {conflicts} URL conflicts)")
    
    return resolved


def apply_label_strategy(label: int, strategy: str) -> int:
    """Map a ViFactCheck label to the target format based on strategy."""
    if strategy == "binary_exclude_nei":
        if label == 0:
            return 0  # Supported → Real
        elif label == 1:
            return 1  # Refuted → Fake
        else:
            return None  # NEI → exclude
    elif strategy == "binary_nei_as_real":
        if label == 0 or label == 2:
            return 0  # Supported/NEI → Real
        elif label == 1:
            return 1  # Refuted → Fake
        else:
            return None
    elif strategy == "three_class":
        if label in (0, 1, 2):
            return label
        else:
            return None
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def inject_labels(
    json_path: Path,
    url_label_map: dict,
    strategy: str,
) -> tuple:
    """
    Inject labels into crawled JSON file.
    
    Returns:
        (labeled_articles, stats_dict)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    labeled = []
    stats = Counter()

    for article in articles:
        source_url = article.get("source_url", "")
        norm_url = normalize_url(source_url)

        raw_label = url_label_map.get(norm_url)

        if raw_label is None:
            stats["no_match"] += 1
            continue

        mapped_label = apply_label_strategy(raw_label, strategy)

        if mapped_label is None:
            stats["excluded_by_strategy"] += 1
            continue

        article["label"] = mapped_label
        labeled.append(article)
        stats[f"label_{mapped_label}"] += 1

    stats["total_input"] = len(articles)
    stats["total_labeled"] = len(labeled)

    return labeled, dict(stats)


def main():
    parser = argparse.ArgumentParser(description="Recover ViFactCheck labels")
    parser.add_argument(
        "--strategy",
        choices=["binary_exclude_nei", "binary_nei_as_real", "three_class"],
        default="binary_exclude_nei",
        help="Label mapping strategy (default: binary_exclude_nei)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./data/json",
        help="Directory containing crawled JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as input-dir, overwrites with _labeled suffix)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats without writing files",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="tranthaihoa/vifactcheck",
        help="HuggingFace dataset name",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "dev", "test"]

    print(f"Strategy: {args.strategy}")
    print(f"Input:    {input_dir}")
    print(f"Output:   {output_dir}")
    print(f"Dry run:  {args.dry_run}")
    print()

    all_stats = {}

    for split in splits:
        print(f"=== Processing {split} ===")

        # Load HF labels
        url_label_map = load_hf_labels(split, args.dataset_name)

        # Find crawled JSON
        json_file = input_dir / f"news_data_vifactcheck_{split}_cleaned.json"
        if not json_file.exists():
            # Try without _cleaned suffix
            json_file = input_dir / f"news_data_vifactcheck_{split}.json"
        if not json_file.exists():
            print(f"  ⚠️  JSON not found: {json_file}, skipping")
            continue

        # Inject labels
        labeled_articles, stats = inject_labels(json_file, url_label_map, args.strategy)

        # Report stats
        print(f"  Input articles:       {stats.get('total_input', 0)}")
        print(f"  Labeled (output):     {stats.get('total_labeled', 0)}")
        print(f"  No URL match:         {stats.get('no_match', 0)}")
        print(f"  Excluded by strategy: {stats.get('excluded_by_strategy', 0)}")
        label_keys = sorted([k for k in stats if k.startswith("label_")])
        for k in label_keys:
            print(f"  {k}: {stats[k]}")

        all_stats[split] = stats

        # Save
        if not args.dry_run:
            out_file = output_dir / f"news_data_vifactcheck_{split}_labeled.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(labeled_articles, f, ensure_ascii=False, indent=2)
            print(f"  ✅ Saved {len(labeled_articles)} articles → {out_file.name}")
        print()

    # Summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for split, stats in all_stats.items():
        total_in = stats.get("total_input", 0)
        total_out = stats.get("total_labeled", 0)
        pct = (total_out / total_in * 100) if total_in > 0 else 0
        label_dist = {k: v for k, v in stats.items() if k.startswith("label_")}
        print(f"  {split}: {total_out}/{total_in} ({pct:.1f}%) — {label_dist}")


if __name__ == "__main__":
    main()
