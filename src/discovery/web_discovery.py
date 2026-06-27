"""
Optional web source discovery — DuckDuckGo default, Tavily/Brave when API key set.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

BLOCKED_DOMAINS = {
    "facebook.com", "twitter.com", "x.com", "instagram.com", "tiktok.com",
    "reddit.com", "pinterest.com", "linkedin.com",
}


def infer_topic(notebook_name: str, source_titles: List[str]) -> str:
    """Build a search query from notebook name and existing sources."""
    parts = [notebook_name.strip()]
    for title in source_titles[:5]:
        t = title.strip()
        if t and t not in parts:
            parts.append(t)
    query = " ".join(parts)
    query = re.sub(r"\s+", " ", query).strip()
    if len(query) > 120:
        query = query[:120].rsplit(" ", 1)[0]
    return query or notebook_name or "research topic"


def _domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def _filter_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for c in candidates:
        url = c.get("url", "").strip()
        if not url or not url.startswith("http"):
            continue
        dom = _domain(url)
        if any(dom == b or dom.endswith("." + b) for b in BLOCKED_DOMAINS):
            continue
        key = url.split("#")[0].rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        c["domain"] = dom
        out.append(c)
    return out


def _search_duckduckgo(query: str, max_results: int) -> List[Dict[str, Any]]:
    try:
        from duckduckgo_search import DDGS

        results: List[Dict[str, Any]] = []
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=max_results):
                results.append({
                    "url": item.get("href") or item.get("link") or "",
                    "title": item.get("title") or "Untitled",
                    "snippet": item.get("body") or item.get("snippet") or "",
                })
        return results
    except ImportError:
        logger.warning("duckduckgo-search not installed")
        return []
    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {e}")
        return []


def _search_tavily(query: str, max_results: int, api_key: str) -> List[Dict[str, Any]]:
    try:
        r = httpx.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "max_results": max_results},
            timeout=30.0,
        )
        r.raise_for_status()
        data = r.json()
        return [
            {
                "url": item.get("url", ""),
                "title": item.get("title") or "Untitled",
                "snippet": item.get("content") or item.get("snippet") or "",
            }
            for item in data.get("results", [])
        ]
    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        return []


def _search_brave(query: str, max_results: int, api_key: str) -> List[Dict[str, Any]]:
    try:
        r = httpx.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": max_results},
            headers={"Accept": "application/json", "X-Subscription-Token": api_key},
            timeout=30.0,
        )
        r.raise_for_status()
        data = r.json()
        return [
            {
                "url": item.get("url", ""),
                "title": item.get("title") or "Untitled",
                "snippet": item.get("description") or "",
            }
            for item in data.get("web", {}).get("results", [])
        ]
    except Exception as e:
        logger.error(f"Brave search failed: {e}")
        return []


def search_candidates(
    query: str,
    provider: str = "duckduckgo",
    max_results: int = 8,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Search the web and return filtered candidate sources."""
    key = api_key or os.getenv("DISCOVER_API_KEY", "")
    provider = (provider or "duckduckgo").lower()

    if provider == "tavily" and key:
        raw = _search_tavily(query, max_results, key)
    elif provider == "brave" and key:
        raw = _search_brave(query, max_results, key)
    else:
        raw = _search_duckduckgo(query, max_results)

    return _filter_candidates(raw)[:max_results]
