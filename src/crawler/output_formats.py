from typing import Dict, Any, Callable, List
from datetime import datetime
import hashlib
from urllib.parse import urlparse
import re

class OutputFormatter:
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean text by removing extra whitespace and normalizing spaces"""
        if not text:
            return ""
        # Remove extra whitespace and normalize spaces
        text = re.sub(r'\s+', ' ', text.strip())
        return text

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract domain from URL"""
        try:
            return urlparse(url).netloc
        except:
            return ""

    @staticmethod
    def _generate_article_id(url: str, title: str) -> str:
        """Generate a unique article ID based on URL and title"""
        combined = f"{url}_{title}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    @staticmethod
    def _process_images(images: List[Dict]) -> List[Dict]:
        """Process and standardize image information"""
        processed = []
        for img in (images or []):
            if not img:
                continue
            processed.append({
                "path": img.get("folder_path", ""),
                "url": img.get("src_url", ""),
                "caption": img.get("caption", ""),
                "alt_text": img.get("alt_text", "")
            })
        return processed

    @staticmethod
    def default_format(result: Any) -> Dict:
        """Default output format"""
        return {
            "title": result.title,
            "content": result.content_text,
            "source_url": result.url,
            "other_urls": result.links,
            "images": result.images,
            "contents": result.contents,
        }

    @staticmethod
    def custom_format(result: Any) -> Dict:
        """Custom output format"""
        return {
            "title": result.title,
            "content": result.content_text,
            "source_url": result.url,
            "other_urls": result.links,
            "images": result.images,
            "contents": result.contents,
            "markdown": result.markdown,
        }

    @staticmethod
    def detailed_format(result: Any) -> Dict:
        """Detailed output with metadata"""
        return {
            "article": {
                "title": result.title,
                "full_text": result.content_text,
                "paragraphs": result.contents
            },
            "media": {
                "images": result.images,
            },
            "urls": {
                "source": result.url,
                "related": result.links
            },
            "metadata": {
                "crawler_success": result.success,
                "error": result.error if not result.success else None,
                "timestamp": datetime.now().isoformat(),
                "language": "vi"
            }
        }

    @staticmethod
    def minimal_format(result: Any) -> Dict:
        """Minimal output with just essential fields"""
        return {
            "title": result.title,
            "text": result.content_text,
            "url": result.url
        }

    @staticmethod
    def ml_training_format(result: Any) -> Dict:
        """Format optimized for machine learning training datasets"""
        clean_title = OutputFormatter._clean_text(result.title)
        clean_text = OutputFormatter._clean_text(result.content_text)
        
        return {
            "id": OutputFormatter._generate_article_id(result.url, clean_title),
            "features": {
                "title": clean_title,
                "text": clean_text,
                "text_length": len(clean_text),
                "paragraph_count": len(result.contents),
                "image_count": len(result.images or []),
                "has_images": bool(result.images),
                "domain": OutputFormatter._extract_domain(result.url),
                "link_count": len(result.links or [])
            },
            "metadata": {
                "source_url": result.url,
                "crawl_timestamp": datetime.now().isoformat(),
                "language": "vi",
                "success": result.success
            },
            "media": {
                "images": OutputFormatter._process_images(result.images)
            }
        }

    @staticmethod
    def content_analysis_format(result: Any) -> Dict:
        """Format optimized for content analysis and fact-checking"""
        domain = OutputFormatter._extract_domain(result.url)
        clean_title = OutputFormatter._clean_text(result.title)
        
        return {
            "article_id": OutputFormatter._generate_article_id(result.url, clean_title),
            "source": {
                "domain": domain,
                "url": result.url,
                "timestamp": datetime.now().isoformat()
            },
            "content": {
                "headline": clean_title,
                "paragraphs": [OutputFormatter._clean_text(p) for p in result.contents if p],
                "full_text": OutputFormatter._clean_text(result.content_text)
            },
            "media_elements": {
                "images": OutputFormatter._process_images(result.images),
                "image_count": len(result.images or [])
            },
            "citations": {
                "linked_articles": result.links,
                "link_count": len(result.links or [])
            },
            "analysis_metadata": {
                "text_statistics": {
                    "paragraph_count": len(result.contents),
                    "total_length": len(result.content_text),
                    "avg_paragraph_length": len(result.content_text) / len(result.contents) if result.contents else 0
                },
                "crawl_status": {
                    "success": result.success,
                    "error": result.error if not result.success else None
                }
            }
        }

    @staticmethod
    def research_format(result: Any) -> Dict:
        """Format optimized for academic research and analysis"""
        domain = OutputFormatter._extract_domain(result.url)
        clean_title = OutputFormatter._clean_text(result.title)
        timestamp = datetime.now()
        
        return {
            "document_metadata": {
                "id": OutputFormatter._generate_article_id(result.url, clean_title),
                "source_domain": domain,
                "url": result.url,
                "collection_date": timestamp.date().isoformat(),
                "collection_time": timestamp.time().isoformat()
            },
            "article_content": {
                "title": clean_title,
                "body": OutputFormatter._clean_text(result.content_text),
                "structured_content": {
                    "paragraphs": [OutputFormatter._clean_text(p) for p in result.contents if p],
                    "paragraph_count": len(result.contents)
                }
            },
            "multimedia_content": {
                "images": OutputFormatter._process_images(result.images),
                "media_statistics": {
                    "total_images": len(result.images or []),
                    "images_with_captions": len([img for img in (result.images or []) if img.get("caption")])
                }
            },
            "reference_data": {
                "outbound_links": result.links,
                "link_domains": [OutputFormatter._extract_domain(link) for link in (result.links or [])]
            },
            "technical_metadata": {
                "language": "vi",
                "crawl_success": result.success,
                "error_info": result.error if not result.success else None,
                "text_metrics": {
                    "total_length": len(result.content_text),
                    "title_length": len(clean_title),
                    "avg_paragraph_length": len(result.content_text) / len(result.contents) if result.contents else 0
                }
            }
        }

    # Register all format functions
    FORMAT_REGISTRY = {
        "default": default_format,
        "custom": custom_format,
        "detailed": detailed_format,
        "minimal": minimal_format,
        "ml_training": ml_training_format,
        "content_analysis": content_analysis_format,
        "research": research_format
    }

    @classmethod
    def get_formatter(cls, format_name: str) -> Callable:
        """Get a formatter function by name"""
        return cls.FORMAT_REGISTRY.get(format_name, cls.default_format)

    @classmethod
    def register_format(cls, name: str, formatter: Callable):
        """Register a new format function"""
        cls.FORMAT_REGISTRY[name] = formatter