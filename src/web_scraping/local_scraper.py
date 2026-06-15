"""
Web Scraper — BeautifulSoup-based content extraction.

Scrapes web pages using urllib + BeautifulSoup. Extracts clean text
content and converts it into DocumentChunks for the RAG pipeline.
"""

import re
import ssl
import logging
import urllib.request
from typing import List, Dict, Any
from urllib.parse import urlparse
from datetime import datetime

from src.document_processing.doc_processor import DocumentChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebScraper:
    """Scrape web pages and convert to DocumentChunks."""

    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

    def scrape_url(
        self,
        url: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> List[DocumentChunk]:
        """Scrape a URL and return DocumentChunks."""

        if not self._is_valid_url(url):
            raise ValueError(f"Invalid URL: {url}")

        logger.info(f"Scraping: {url}")

        try:
            from bs4 import BeautifulSoup

            # Use urllib (not httpx) — compatible with sites like Wikipedia
            # that block httpx's TLS fingerprint
            req = urllib.request.Request(url, headers={
                "User-Agent": self.USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            })

            # Create SSL context that doesn't verify (for sites with bad certs)
            ctx = ssl.create_default_context()
            response = urllib.request.urlopen(req, timeout=30, context=ctx)
            html = response.read().decode("utf-8", errors="ignore")

            soup = BeautifulSoup(html, "html.parser")

            # Remove noise elements
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
                tag.decompose()

            # Find main content area
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
            content = re.sub(r"\n{3,}", "\n\n", content)  # Clean whitespace

            if not content.strip():
                return []

            # Create chunks
            chunks = self._create_chunks(content, title, url, chunk_size, chunk_overlap)
            logger.info(f"Scraped {url}: {len(chunks)} chunks, title: {title}")
            return chunks

        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            raise

    def _create_chunks(
        self, content: str, title: str, url: str,
        chunk_size: int, chunk_overlap: int,
    ) -> List[DocumentChunk]:
        """Split content into DocumentChunks."""
        chunks = []
        start = 0
        idx = 0

        while start < len(content):
            end = min(start + chunk_size, len(content))

            # Break at natural boundaries
            if end < len(content):
                for sep in ["\n\n", ". ", "\n"]:
                    pos = content.rfind(sep, start, end)
                    if pos > start + chunk_size * 0.4:
                        end = pos + len(sep)
                        break

            text = content[start:end].strip()
            if text:
                chunks.append(DocumentChunk(
                    content=text,
                    source_file=title,
                    source_type="web",
                    chunk_index=idx,
                    start_char=start,
                    end_char=end - 1,
                    metadata={
                        "url": url,
                        "domain": urlparse(url).netloc,
                        "scraped_at": datetime.now().isoformat(),
                    },
                ))
                idx += 1

            start = end if end >= start + chunk_size else max(end, start + chunk_size - chunk_overlap)

        return chunks

    def _is_valid_url(self, url: str) -> bool:
        try:
            result = urlparse(url)
            return all([result.scheme in ("http", "https"), result.netloc])
        except Exception:
            return False
