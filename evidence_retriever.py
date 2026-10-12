"""Multi-source Evidence Retriever with:
1. Google Fact Check Tools API (PIB, Alt News, BOOM, Factly)
2. Live Web Search (Tavily / Brave / SerpAPI / DDGS fallback) with domain allow-list boosting:
   pib.gov.in, factcheck.pib.gov.in, who.int, mohfw.gov.in, icmr.gov.in, rbi.org.in,
   india.gov.in, myscheme.gov.in, data.gov.in, eci.gov.in, isro.gov.in, wikipedia.org,
   reuters.com, thehindu.com, altnews.in, boomlive.in, factly.in, vishvasnews.com
3. Wikipedia REST API entity lookups
4. Actual page content retrieval using Trafilatura (ad-free clean text)
5. 24-hour SQL caching to protect API quotas
"""
import re
import json
import logging
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import httpx
import trafilatura
from sqlalchemy import select

from backend.app.core.config import settings
from backend.app.core.security import is_ssrf_safe_url
from backend.app.core.database import AsyncSessionLocal, SearchCache

logger = logging.getLogger("sach.retrieval")

TRUSTED_DOMAINS = [
    "pib.gov.in",
    "factcheck.pib.gov.in",
    "who.int",
    "mohfw.gov.in",
    "icmr.gov.in",
    "rbi.org.in",
    "india.gov.in",
    "myscheme.gov.in",
    "data.gov.in",
    "eci.gov.in",
    "isro.gov.in",
    "wikipedia.org",
    "reuters.com",
    "thehindu.com",
    "altnews.in",
    "boomlive.in",
    "factly.in",
    "vishvasnews.com",
    "bbc.com",
    "ndtv.com",
    "theprint.in"
]


class RetrievedSource:
    def __init__(self, title: str, publisher: str, url: str, snippet: str, date: Optional[str] = None):
        self.title = title
        self.publisher = publisher
        self.url = url
        self.snippet = snippet
        self.date = date
        self.page_text: Optional[str] = None
        self.is_trusted_domain = any(domain in url.lower() for domain in TRUSTED_DOMAINS)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "publisher": self.publisher,
            "url": self.url,
            "snippet": self.snippet,
            "date": self.date,
            "page_text": self.page_text,
            "is_trusted_domain": self.is_trusted_domain
        }


def extract_publisher_from_url(url: str) -> str:
    """Derives a human-friendly publisher name from the domain."""
    for domain in TRUSTED_DOMAINS:
        if domain in url.lower():
            if "pib.gov.in" in domain:
                return "Press Information Bureau (PIB)"
            elif "who.int" in domain:
                return "World Health Organization (WHO)"
            elif "mohfw.gov.in" in domain:
                return "Ministry of Health & Family Welfare"
            elif "rbi.org.in" in domain:
                return "Reserve Bank of India (RBI)"
            elif "wikipedia.org" in domain:
                return "Wikipedia"
            elif "altnews.in" in domain:
                return "Alt News"
            elif "boomlive.in" in domain:
                return "BOOM Live"
            elif "factly.in" in domain:
                return "Factly"
            elif "vishvasnews.com" in domain:
                return "Vishvas News"
            elif "reuters.com" in domain:
                return "Reuters"
            elif "thehindu.com" in domain:
                return "The Hindu"
            elif "myscheme.gov.in" in domain:
                return "myScheme (Govt of India)"

    # Fallback to domain host
    try:
        from urllib.parse import urlparse
        host = urlparse(url).netloc.replace("www.", "")
        return host.split(".")[0].capitalize()
    except Exception:
        return "Web Source"


async def get_cached_search(query_key: str) -> Optional[List[Dict[str, Any]]]:
    """Retrieves cached search items if within 24 hours."""
    try:
        async with AsyncSessionLocal() as session:
            stmt = select(SearchCache).where(SearchCache.query_key == query_key)
            result = await session.execute(stmt)
            cache_row = result.scalars().first()
            if cache_row and not cache_row.is_expired(ttl_hours=settings.CACHE_TTL_HOURS):
                return json.loads(cache_row.results_json)
    except Exception as e:
        logger.warning(f"Cache read error: {e}")
    return None


async def save_cached_search(query_key: str, results: List[Dict[str, Any]]):
    """Saves search results to SQL cache with 24-hour TTL."""
    try:
        async with AsyncSessionLocal() as session:
            stmt = select(SearchCache).where(SearchCache.query_key == query_key)
            result = await session.execute(stmt)
            cache_row = result.scalars().first()
            if cache_row:
                cache_row.results_json = json.dumps(results)
                cache_row.created_at = datetime.now(timezone.utc)
            else:
                new_cache = SearchCache(
                    query_key=query_key,
                    results_json=json.dumps(results),
                    created_at=datetime.now(timezone.utc)
                )
                session.add(new_cache)
            await session.commit()
    except Exception as e:
        logger.warning(f"Cache write error: {e}")


async def search_google_fact_check(query: str) -> List[RetrievedSource]:
    """Queries Google Fact Check Tools API for verified fact checks."""
    if not settings.GOOGLE_FACTCHECK_API_KEY:
        return []

    sources = []
    try:
        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {
            "query": query,
            "key": settings.GOOGLE_FACTCHECK_API_KEY,
            "pageSize": 5,
        }
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            if resp.status_code == 200:
                data = resp.json()
                for claim in data.get("claims", []):
                    for review in claim.get("claimReview", []):
                        review_url = review.get("url")
                        if review_url:
                            safe, _ = is_ssrf_safe_url(review_url)
                            if safe:
                                publisher = review.get("publisher", {}).get("name", "Fact Checker")
                                title = review.get("title") or claim.get("text", "Fact Check Review")
                                textual_rating = review.get("textualRating", "")
                                snippet = f"Rating: {textual_rating}. Claim: {claim.get('text', '')}"
                                date = review.get("reviewDate")
                                sources.append(RetrievedSource(
                                    title=title,
                                    publisher=publisher,
                                    url=review_url,
                                    snippet=snippet,
                                    date=date
                                ))
    except Exception as e:
        logger.warning(f"Google Fact Check API query failed: {e}")

    return sources


async def search_tavily(query: str) -> List[RetrievedSource]:
    """Queries Tavily API for fresh evidence and fact checks."""
    if not settings.TAVILY_API_KEY:
        return []

    sources = []
    try:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": settings.TAVILY_API_KEY,
            "query": query,
            "search_depth": "advanced",
            "include_domains": TRUSTED_DOMAINS[:10],
            "max_results": 5
        }
        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("results", []):
                    item_url = item.get("url")
                    safe, _ = is_ssrf_safe_url(item_url)
                    if safe:
                        sources.append(RetrievedSource(
                            title=item.get("title", "News / Report"),
                            publisher=extract_publisher_from_url(item_url),
                            url=item_url,
                            snippet=item.get("content", ""),
                            date=item.get("published_date")
                        ))
    except Exception as e:
        logger.warning(f"Tavily search failed: {e}")

    return sources


async def search_brave(query: str) -> List[RetrievedSource]:
    """Queries Brave Search API."""
    if not settings.BRAVE_API_KEY:
        return []

    sources = []
    try:
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {"X-Subscription-Token": settings.BRAVE_API_KEY}
        params = {"q": query, "count": 5}
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=headers, params=params)
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("web", {}).get("results", []):
                    item_url = item.get("url")
                    safe, _ = is_ssrf_safe_url(item_url)
                    if safe:
                        sources.append(RetrievedSource(
                            title=item.get("title", ""),
                            publisher=extract_publisher_from_url(item_url),
                            url=item_url,
                            snippet=item.get("description", ""),
                            date=item.get("page_age")
                        ))
    except Exception as e:
        logger.warning(f"Brave search failed: {e}")

    return sources


async def search_wikipedia(query: str) -> List[RetrievedSource]:
    """Searches Wikipedia REST API for background entity knowledge."""
    sources = []
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "opensearch",
            "search": query,
            "limit": 3,
            "namespace": 0,
            "format": "json"
        }
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(url, params=params)
            if resp.status_code == 200:
                data = resp.json()
                if len(data) >= 4 and len(data[1]) > 0:
                    titles = data[1]
                    snippets = data[2]
                    urls = data[3]
                    for title, snip, u in zip(titles, snippets, urls):
                        safe, _ = is_ssrf_safe_url(u)
                        if safe and snip:
                            sources.append(RetrievedSource(
                                title=title,
                                publisher="Wikipedia",
                                url=u,
                                snippet=snip,
                                date=None
                            ))
    except Exception as e:
        logger.warning(f"Wikipedia search failed: {e}")

    return sources


async def search_ddgs_fallback(query: str) -> List[RetrievedSource]:
    """Fallback search using DDGS."""
    sources = []
    try:
        from ddgs import DDGS
        # Run blocking ddgs in thread pool
        def _fetch():
            ddgs = DDGS()
            # Boost trusted domains by searching both direct and with site filters
            results = list(ddgs.text(f"{query} fact check OR PIB OR RBI", max_results=5))
            if not results:
                results = list(ddgs.text(query, max_results=5))
            return results

        raw_results = await asyncio.to_thread(_fetch)
        for item in raw_results:
            u = item.get("href")
            if u:
                safe, _ = is_ssrf_safe_url(u)
                if safe:
                    sources.append(RetrievedSource(
                        title=item.get("title", ""),
                        publisher=extract_publisher_from_url(u),
                        url=u,
                        snippet=item.get("body", ""),
                        date=None
                    ))
    except Exception as e:
        logger.warning(f"DDGS search failed: {e}")

    return sources


async def fetch_page_trafilatura(url: str) -> Optional[str]:
    """Fetches the actual page text using Trafilatura for genuine LLM grounding."""
    safe, _ = is_ssrf_safe_url(url)
    if not safe:
        return None

    def _fetch_sync():
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(
                    downloaded,
                    include_comments=False,
                    include_tables=True,
                    no_fallback=False
                )
                if text:
                    # Return top 2500 characters of clean parsed article text
                    return text[:2500]
        except Exception:
            pass
        return None

    return await asyncio.to_thread(_fetch_sync)


async def retrieve_evidence_for_claim(query: str) -> List[RetrievedSource]:
    """Full retrieval pipeline for a single normalized query:
    1. Check SQL Cache (24h TTL)
    2. Fact Check API + Trusted domain web search
    3. Wikipedia background check
    4. Fetch real page texts with Trafilatura
    5. Rank by source trust + relevance
    6. Save to cache
    """
    cache_key = query.strip().lower()
    cached = await get_cached_search(cache_key)
    if cached:
        logger.info(f"Loaded {len(cached)} evidence items from 24h cache for query: {query}")
        reconstructed = []
        for d in cached:
            s = RetrievedSource(
                title=d["title"],
                publisher=d["publisher"],
                url=d["url"],
                snippet=d["snippet"],
                date=d.get("date")
            )
            s.page_text = d.get("page_text")
            s.is_trusted_domain = d.get("is_trusted_domain", False)
            reconstructed.append(s)
        return reconstructed

    # Execute search providers concurrently
    factcheck_task = search_google_fact_check(query)
    tavily_task = search_tavily(query)
    brave_task = search_brave(query)
    wiki_task = search_wikipedia(query)
    ddgs_task = search_ddgs_fallback(query)

    gathered = await asyncio.gather(
        factcheck_task,
        tavily_task,
        brave_task,
        wiki_task,
        ddgs_task,
        return_exceptions=True
    )

    combined_sources: List[RetrievedSource] = []
    seen_urls = set()

    for res in gathered:
        if isinstance(res, list):
            for src in res:
                if src.url not in seen_urls:
                    seen_urls.add(src.url)
                    combined_sources.append(src)

    # Sort sources: trusted domains first, then length of snippet
    combined_sources.sort(key=lambda x: (1 if x.is_trusted_domain else 0, len(x.snippet)), reverse=True)
    top_sources = combined_sources[:5]

    # Concurrently fetch actual page content with Trafilatura for top sources
    async def _fill_page(src: RetrievedSource):
        page_content = await fetch_page_trafilatura(src.url)
        if page_content:
            src.page_text = page_content
        else:
            src.page_text = src.snippet

    if top_sources:
        await asyncio.gather(*[_fill_page(s) for s in top_sources], return_exceptions=True)

    # Cache results
    serialized = [s.to_dict() for s in top_sources]
    await save_cached_search(cache_key, serialized)

    return top_sources
