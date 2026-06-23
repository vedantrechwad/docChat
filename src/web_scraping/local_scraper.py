"""
Web Scraper — BeautifulSoup-based content extraction.
"""

import re
import ssl
import logging
import urllib.request
from typing import List, Optional
from urllib.parse import urlparse
from datetime import datetime

from src.document_processing.document_chunk import DocumentChunk
from src.document_processing.chunking_service import ChunkingService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebScraper:
    """Scrape web pages and convert to DocumentChunks."""

    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

    def __init__(self, chunking: Optional[ChunkingService] = None):
        self.chunking = chunking or ChunkingService.from_preset("balanced")

    def set_chunking(self, chunking: ChunkingService) -> None:
        self.chunking = chunking

    def scrape_url(self, url: str) -> List[DocumentChunk]:
        """Scrape a URL and return DocumentChunks."""
        if not self._is_valid_url(url):
            raise ValueError(f"Invalid URL: {url}")

        logger.info(f"Scraping: {url}")

        try:
            from bs4 import BeautifulSoup

            req = urllib.request.Request(url, headers={
                "User-Agent": self.USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            })

            ctx = ssl.create_default_context()
            response = urllib.request.urlopen(req, timeout=30, context=ctx)
            html = response.read().decode("utf-8", errors="ignore")

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
                logger.warning(f"No content found on {url}")
                return []

            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else urlparse(url).netloc
            content = main.get_text(separator="\n", strip=True)
            content = re.sub(r"\n{3,}", "\n\n", content)

            if not content.strip():
                return []

            chunks = self.chunking.create_chunks(
                text=content,
                source_file=title,
                source_type="web",
                additional_metadata={
                    "url": url,
                    "domain": urlparse(url).netloc,
                    "scraped_at": datetime.now().isoformat(),
                },
            )
            logger.info(f"Scraped {url}: {len(chunks)} chunks, title: {title}")
            return chunks

        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            raise

    def _is_valid_url(self, url: str) -> bool:
        try:
            result = urlparse(url)
            return all([result.scheme in ("http", "https"), result.netloc])
        except Exception:
            return False
