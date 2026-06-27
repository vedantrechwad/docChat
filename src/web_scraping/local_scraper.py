"""
Web Scraper — trafilatura + BeautifulSoup content extraction.
"""

import hashlib
import re
import ssl
import logging
import urllib.request
from typing import List, Optional, Tuple
from urllib.parse import urlparse
from datetime import datetime

from src.document_processing.document_chunk import DocumentChunk
from src.document_processing.chunking_service import ChunkingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebScraper:
    """Scrape web pages and convert to DocumentChunks."""

    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )

    def __init__(self, chunking: Optional[ChunkingService] = None):
        self.chunking = chunking or ChunkingService.from_preset("balanced")

    def set_chunking(self, chunking: ChunkingService) -> None:
        self.chunking = chunking

    @staticmethod
    def url_source_name(url: str) -> str:
        """Stable source name from URL (avoids title collisions)."""
        parsed = urlparse(url)
        host = parsed.netloc.replace("www.", "")
        path = parsed.path.strip("/").replace("/", "_")[:40]
        digest = hashlib.sha256(url.encode()).hexdigest()[:8]
        label = f"{host}/{path}" if path else host
        return f"web:{label}#{digest}"

    def scrape_url(self, url: str) -> List[DocumentChunk]:
        """Scrape a URL and return DocumentChunks."""
        if not self._is_valid_url(url):
            raise ValueError(f"Invalid URL: {url}")

        logger.info(f"Scraping: {url}")

        try:
            html, title = self._fetch_html(url)
            content = self._extract_content(html, url)

            if not content.strip():
                raise ValueError(
                    "No readable content on this page. The site may require JavaScript, "
                    "be paywalled, or block automated access."
                )

            source_name = self.url_source_name(url)
            chunks = self.chunking.create_chunks(
                text=content,
                source_file=source_name,
                source_type="web",
                additional_metadata={
                    "url": url,
                    "title": title,
                    "domain": urlparse(url).netloc,
                    "scraped_at": datetime.now().isoformat(),
                },
            )
            logger.info(f"Scraped {url}: {len(chunks)} chunks, title: {title}")
            return chunks

        except ValueError:
            raise
        except urllib.error.HTTPError as e:
            raise ValueError(f"HTTP {e.code}: page returned an error ({url})") from e
        except urllib.error.URLError as e:
            reason = getattr(e, "reason", e)
            if "timed out" in str(reason).lower():
                raise ValueError(f"Request timed out after 30s: {url}") from e
            if "ssl" in str(reason).lower():
                raise ValueError(f"SSL certificate error for {url}") from e
            raise ValueError(f"Could not reach URL: {reason}") from e
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            raise ValueError(f"Failed to scrape URL: {e}") from e

    def _fetch_html(self, url: str) -> Tuple[str, str]:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": self.USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        ctx = ssl.create_default_context()
        response = urllib.request.urlopen(req, timeout=30, context=ctx)
        html = response.read().decode("utf-8", errors="ignore")
        title = self._extract_title(html, url)
        return html, title

    def _extract_title(self, html: str, url: str) -> str:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            title_tag = soup.find("title")
            if title_tag:
                return title_tag.get_text(strip=True)
        except Exception:
            pass
        return urlparse(url).netloc

    def _extract_content(self, html: str, url: str) -> str:
        content = self._extract_with_trafilatura(html, url)
        if content and len(content.strip()) >= 80:
            return content
        return self._extract_with_beautifulsoup(html)

    def _extract_with_trafilatura(self, html: str, url: str) -> str:
        try:
            import trafilatura
            text = trafilatura.extract(
                html,
                url=url,
                include_comments=False,
                include_tables=True,
                favor_precision=True,
            )
            if text:
                return re.sub(r"\n{3,}", "\n\n", text.strip())
        except ImportError:
            logger.debug("trafilatura not installed, using BeautifulSoup fallback")
        except Exception as e:
            logger.warning(f"trafilatura extraction failed for {url}: {e}")
        return ""

    def _extract_with_beautifulsoup(self, html: str) -> str:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
            tag.decompose()

        main = (
            soup.find("main")
            or soup.find("article")
            or soup.find(attrs={"role": "main"})
            or soup.find(class_="content")
            or soup.body
        )
        if not main:
            return ""

        content = main.get_text(separator="\n", strip=True)
        return re.sub(r"\n{3,}", "\n\n", content)

    def _is_valid_url(self, url: str) -> bool:
        try:
            result = urlparse(url)
            return all([result.scheme in ("http", "https"), result.netloc])
        except Exception:
            return False
