from pathlib import Path

# Ignore training-pipeline tests that require src/ files not present in this
# repo checkout (pair_extractor.py lives in the training source tree).
collect_ignore = [
    str(Path(__file__).parent / "processing" / "coolant" / "test_pair_extractor.py"),
]
