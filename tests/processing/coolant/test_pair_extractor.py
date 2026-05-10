"""
Tests for PairExtractor with metadata and stats support.
"""

import json
import tempfile
from pathlib import Path
import pytest
import importlib.util

# Load pair_extractor module directly
pair_extractor_path = Path(__file__).parent.parent.parent.parent / "src" / "processing" / "coolant" / "pair_extractor.py"
spec = importlib.util.spec_from_file_location("pair_extractor", pair_extractor_path)
pair_extractor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pair_extractor_module)
PairExtractor = pair_extractor_module.PairExtractor


@pytest.fixture
def temp_json_and_images():
    """Create temporary JSON file and image directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        jpg_dir = tmpdir / "jpg"
        jpg_dir.mkdir()
        
        # Create dummy image files
        (jpg_dir / "source1").mkdir()
        (jpg_dir / "source1" / "img1.jpg").write_bytes(b"fake image data")
        (jpg_dir / "source1" / "img2.jpg").write_bytes(b"fake image data")
        (jpg_dir / "source2").mkdir()
        (jpg_dir / "source2" / "img3.jpg").write_bytes(b"fake image data")
        
        # Create test JSON with articles and images
        articles = [
            {
                "title": "Article One",
                "label": "real",
                "source_url": "https://example.com/article1",
                "images": [
                    {
                        "caption": "A beautiful sunset",
                        "folder_path": "source1/img1.jpg"
                    },
                    {
                        "caption": "Ảnh: NVCC",  # Credit only, should be skipped
                        "folder_path": "source1/img2.jpg"
                    }
                ]
            },
            {
                "title": "Article Two",
                "label": "fake",
                "source_url": "https://example.com/article2",
                "images": [
                    {
                        "caption": "Mountain landscape view",
                        "folder_path": "source2/img3.jpg"
                    }
                ]
            }
        ]
        
        json_path = tmpdir / "test_data.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False)
        
        yield json_path, jpg_dir


def test_extract_from_json_backward_compatible(temp_json_and_images):
    """Test that extract_from_json remains backward compatible (returns list)."""
    json_path, jpg_dir = temp_json_and_images
    
    extractor = PairExtractor(str(jpg_dir), min_caption_len=5)
    result = extractor.extract_from_json(str(json_path))
    
    # Should return a list
    assert isinstance(result, list)
    
    # Should have 2 valid pairs (credit-only caption is skipped)
    assert len(result) == 2
    
    # Check pair structure
    for pair in result:
        assert "image_path" in pair
        assert "caption" in pair
        assert "article_idx" in pair
        assert "folder_path" in pair
        assert "pair_text" in pair
        assert "title" in pair
        assert "source_url" in pair
        assert "source_label" in pair


def test_extract_from_json_return_stats(temp_json_and_images):
    """Test that extract_from_json with return_stats=True returns (pairs, stats) tuple."""
    json_path, jpg_dir = temp_json_and_images
    
    extractor = PairExtractor(str(jpg_dir), min_caption_len=5)
    result = extractor.extract_from_json(str(json_path), return_stats=True)
    
    # Should return a tuple
    assert isinstance(result, tuple)
    assert len(result) == 2
    
    pairs, stats = result
    
    # Check pairs
    assert isinstance(pairs, list)
    assert len(pairs) == 2
    
    # Check stats structure
    assert isinstance(stats, dict)
    assert "raw_articles" in stats
    assert "total_images" in stats
    assert "valid_pairs" in stats
    assert "skipped" in stats
    assert "source_label_counts" in stats
    
    # Check stats values
    assert stats["raw_articles"] == 2
    assert stats["total_images"] == 3
    assert stats["valid_pairs"] == 2
    # The credit-only caption "Ảnh: NVCC" becomes empty after cleaning, so it's counted as no_caption
    assert stats["skipped"]["no_caption"] == 1
    assert stats["source_label_counts"]["real"] == 1
    assert stats["source_label_counts"]["fake"] == 1


def test_pair_text_field(temp_json_and_images):
    """Test that pair_text combines title and caption."""
    json_path, jpg_dir = temp_json_and_images
    
    extractor = PairExtractor(str(jpg_dir), min_caption_len=5)
    pairs = extractor.extract_from_json(str(json_path))
    
    # First pair should have title + caption
    assert pairs[0]["pair_text"] == "Article One A beautiful sunset"
    assert pairs[1]["pair_text"] == "Article Two Mountain landscape view"


def test_source_label_counts(temp_json_and_images):
    """Test that source_label_counts are tracked correctly."""
    json_path, jpg_dir = temp_json_and_images
    
    extractor = PairExtractor(str(jpg_dir), min_caption_len=5)
    pairs, stats = extractor.extract_from_json(str(json_path), return_stats=True)
    
    label_counts = stats["source_label_counts"]
    assert label_counts["real"] == 1
    assert label_counts["fake"] == 1
