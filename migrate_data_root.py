"""
Migrate existing data directories from inside the repo to an external DATA_ROOT.

Usage:
    python migrate_data_root.py /Volumes/MyDrive/fake-news-data

What it moves (repo-relative → new root):
    data/               → <new_root>/data/
    processed_data/     → <new_root>/processed_data/
    training/           → <new_root>/training/
    notebooks/data/     → <new_root>/data/          (merged)
    notebooks/mlruns/   → <new_root>/mlruns/

After migrating, add to your .env:
    DATA_ROOT=/Volumes/MyDrive/fake-news-data
"""

import shutil
import sys
from pathlib import Path


def _move(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"  skip  {src}  (not found)")
        return
    if dst.exists():
        print(f"  skip  {src}  → {dst}  (destination exists)")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    print(f"  moved {src}  → {dst}")


def _merge_dir(src: Path, dst: Path) -> None:
    """Move all children of src into dst (merging, not overwriting)."""
    if not src.exists():
        print(f"  skip  {src}  (not found)")
        return
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        dest_child = dst / child.name
        if dest_child.exists():
            print(f"  skip  {child}  → {dest_child}  (destination exists)")
        else:
            shutil.move(str(child), str(dest_child))
            print(f"  moved {child}  → {dest_child}")
    if not any(src.iterdir()):
        src.rmdir()


def migrate(new_root: Path) -> None:
    repo_root = Path(__file__).parent.resolve()

    if new_root.resolve() == repo_root:
        print("New root is the same as the repo root — nothing to do.")
        return

    print(f"\nMigrating data from:\n  {repo_root}\nto:\n  {new_root}\n")
    new_root.mkdir(parents=True, exist_ok=True)

    _move(repo_root / "data",          new_root / "data")
    _move(repo_root / "processed_data", new_root / "processed_data")
    _move(repo_root / "training",       new_root / "training")

    _merge_dir(repo_root / "notebooks" / "data",    new_root / "data")
    _merge_dir(repo_root / "notebooks" / "mlruns",  new_root / "mlruns")

    print(f"\nDone. Add this to your .env file:\n  DATA_ROOT={new_root}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python migrate_data_root.py <new_data_root_path>")
        sys.exit(1)
    migrate(Path(sys.argv[1]).expanduser().resolve())
