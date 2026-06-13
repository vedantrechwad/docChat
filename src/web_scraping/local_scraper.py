"""
Local Web Scraper - Playwright-based web scraping.

Replaces the paid Firecrawl API with a fully local headless browser
using Playwright. Handles JavaScript-heavy pages robustly.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from urllib.parse import urlparse
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WebPageData:
    """Represents scraped web page data with additional metadata."""
    url: str
    title: str
    content: str
    metadata: Dict[str, Any]
    success: bool
    error: Optional[str] = None


class LocalWebScraper:
    """
    Local web scraper using Playwright for robust JS-rendered page scraping.
    Falls back to httpx + BeautifulSoup for simple pages.
    """

    def __init__(self):
        self._playwright = None
        self._browser = None
        logger.info("LocalWebScraper initialized")

    async def _ensure_browser(self):
        """Lazily initialize the Playwright browser."""
        if self._browser is None:
            try:
                from playwright.async_api import async_playwright
                self._playwright = await async_playwright().start()
                self._browser = await self._playwright.chromium.launch(headless=True)
                logger.info("Playwright browser launched")
            except Exception as e:
                logger.warning(f"Playwright not available ({e}), falling back to httpx+BS4")
                self._browser = None

    def scrape_url(
        self,
        url: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        wait_for_results: int = 30,
    ) -> list:
        """
        Synchronous wrapper for scraping a URL.
        Uses Playwright if available, falls back to httpx + BeautifulSoup.
        """
        if not self._is_valid_url(url):
            raise ValueError(f"Invalid URL format: {url}")

        logger.info(f"Scraping URL: {url}")

        try:
            # Try Playwright first
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                page_data = loop.run_until_complete(
                    self._scrape_with_playwright(url, wait_for_results)
                )
            finally:
                loop.close()
        except Exception as e:
            logger.warning(f"Playwright scraping failed ({e}), trying fallback")
            page_data = self._scrape_with_httpx(url)

        if not page_data.success:
            logger.warning(f"Scraping failed for {url}: {page_data.error}")
            return []

        chunks = self._create_chunks_from_web_content(page_data, chunk_size, chunk_overlap)
        logger.info(f"Successfully scraped {url}: {len(chunks)} chunks created")
        return chunks

    async def _scrape_with_playwright(self, url: str, timeout: int = 30) -> WebPageData:
        """Scrape a page using Playwright (handles JS-rendered content)."""
        await self._ensure_browser()

        if self._browser is None:
            raise RuntimeError("Browser not available")

        page = await self._browser.new_page()
        try:
            await page.goto(url, timeout=timeout * 1000, wait_until="networkidle")

            title = await page.title()

            # Extract clean text content
            content = await page.evaluate("""
                () => {
                    // Remove scripts, styles, nav, footer, ads
                    const removeTags = ['script', 'style', 'nav', 'footer', 'header', 
                                       'aside', 'iframe', 'noscript'];
                    removeTags.forEach(tag => {
                        document.querySelectorAll(tag).forEach(el => el.remove());
                    });
                    
                    // Remove elements with common ad/nav class names
                    const removeClasses = ['ad', 'advertisement', 'sidebar', 'menu', 
                                          'navigation', 'cookie', 'popup', 'modal'];
                    removeClasses.forEach(cls => {
                        document.querySelectorAll(`[class*="${cls}"]`).forEach(el => el.remove());
                    });
                    
                    // Get the main content area or fall back to body
                    const main = document.querySelector('main, article, [role="main"], .content, #content');
                    const targetEl = main || document.body;
                    
                    return targetEl.innerText;
                }
            """)

            # Get page metadata
            description = await page.evaluate("""
                () => {
                    const meta = document.querySelector('meta[name="description"]');
                    return meta ? meta.getAttribute('content') : '';
                }
            """)

            metadata = {
                "scraped_at": datetime.now().isoformat(),
                "original_url": url,
                "title": title,
                "description": description or "",
                "word_count": len(content.split()) if content else 0,
                "character_count": len(content) if content else 0,
                "domain": urlparse(url).netloc,
            }

            return WebPageData(
                url=url,
                title=title or f"Web Page - {urlparse(url).netloc}",
                content=content.strip() if content else "",
                metadata=metadata,
                success=bool(content and content.strip()),
            )

        except Exception as e:
            logger.error(f"Playwright scraping error for {url}: {e}")
            return WebPageData(
                url=url,
                title=f"Error - {urlparse(url).netloc}",
                content="",
                metadata={"error": str(e), "scraped_at": datetime.now().isoformat()},
                success=False,
                error=str(e),
            )
        finally:
            await page.close()

    def _scrape_with_httpx(self, url: str) -> WebPageData:
        """Fallback scraper using httpx + BeautifulSoup (no JS rendering)."""
        try:
            import httpx
            from bs4 import BeautifulSoup

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }

            response = httpx.get(url, headers=headers, timeout=30, follow_redirects=True)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove unwanted elements
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
                tag.decompose()

            # Try to find main content
            main_content = (
                soup.find("main") or
                soup.find("article") or
                soup.find(attrs={"role": "main"}) or
                soup.find(class_="content") or
                soup.body
            )

            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else urlparse(url).netloc

            content = main_content.get_text(separator="\n", strip=True) if main_content else ""

            # Clean up excessive whitespace
            content = re.sub(r"\n{3,}", "\n\n", content)

            desc_tag = soup.find("meta", attrs={"name": "description"})
            description = desc_tag.get("content", "") if desc_tag else ""

            metadata = {
                "scraped_at": datetime.now().isoformat(),
                "original_url": url,
                "title": title,
                "description": description,
                "word_count": len(content.split()) if content else 0,
                "character_count": len(content) if content else 0,
                "domain": urlparse(url).netloc,
            }

            return WebPageData(
                url=url,
                title=title,
                content=content,
                metadata=metadata,
                success=bool(content.strip()),
            )

        except Exception as e:
            logger.error(f"httpx/BS4 scraping error for {url}: {e}")
            return WebPageData(
                url=url,
                title=f"Error - {urlparse(url).netloc}",
                content="",
                metadata={"error": str(e)},
                success=False,
                error=str(e),
            )

    def _create_chunks_from_web_content(
        self,
        page_data: WebPageData,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list:
        """Split scraped content into DocumentChunks."""
        from src.document_processing.doc_processor import DocumentChunk

        if not page_data.success or not page_data.content.strip():
            return []

        chunks = []
        content = page_data.content
        start = 0
        chunk_index = 0

        while start < len(content):
            end = min(start + chunk_size, len(content))

            # Try to break at natural boundaries
            if end < len(content):
                last_double_newline = content.rfind("\n\n", start, end)
                if last_double_newline > start + chunk_size * 0.3:
                    end = last_double_newline + 2
                else:
                    last_period = content.rfind(".", start, end)
                    if last_period > start + chunk_size * 0.5:
                        end = last_period + 1

            chunk_text = content[start:end].strip()

            if chunk_text:
                chunk_metadata = page_data.metadata.copy()
                chunk_metadata.update({
                    "chunk_character_start": start,
                    "chunk_character_end": end - 1,
                    "url_fragment": f"{page_data.url}#chunk-{chunk_index}",
                })

                chunk = DocumentChunk(
                    content=chunk_text,
                    source_file=page_data.title,
                    source_type="web",
                    page_number=None,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end - 1,
                    metadata=chunk_metadata,
                )
                chunks.append(chunk)
                chunk_index += 1

            start = max(start + chunk_size - chunk_overlap, end)

        return chunks

    def batch_scrape_urls(
        self,
        urls: List[str],
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ) -> List[list]:
        """Scrape multiple URLs."""
        all_chunks = []
        for url in urls:
            try:
                chunks = self.scrape_url(url, chunk_size, chunk_overlap)
                all_chunks.append(chunks)
            except Exception as e:
                logger.error(f"Failed to scrape {url}: {e}")
                all_chunks.append([])
        return all_chunks

    def get_url_preview(self, url: str) -> Dict[str, Any]:
        """Get a quick preview of a URL."""
        try:
            page_data = self._scrape_with_httpx(url)
            return {
                "url": url,
                "title": page_data.title,
                "description": page_data.metadata.get("description", ""),
                "word_count": page_data.metadata.get("word_count", 0),
                "domain": urlparse(url).netloc,
                "content_preview": page_data.content[:500] + "..." if len(page_data.content) > 500 else page_data.content,
            }
        except Exception as e:
            return {"error": str(e)}

    def _is_valid_url(self, url: str) -> bool:
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    async def close(self):
        """Clean up browser resources."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
