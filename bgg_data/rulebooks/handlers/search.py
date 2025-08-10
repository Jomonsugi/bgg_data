"""
Web search system using Tavily (primary) with DuckDuckGo (fallback).
Keeps the existing scoring and interface so downstream code remains compatible.
"""

import logging
import time
from typing import List, Dict, Optional, Tuple
from urllib.parse import quote, urlparse, unquote
import base64
from dataclasses import dataclass
from abc import ABC, abstractmethod

import os
import requests
from bs4 import BeautifulSoup

try:
    # Prefer official Tavily client
    from tavily import TavilyClient  # type: ignore
    _TAVILY_AVAILABLE = True
except Exception:
    _TAVILY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Structured search result with metadata."""
    url: str
    title: str
    score: int = 0
    source: str = "unknown"
    confidence: float = 0.0


class SearchProvider(ABC):
    """Abstract base class for search providers."""
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36'
        })
    
    @abstractmethod
    def search(self, query: str) -> List[SearchResult]:
        """Perform a search and return results."""
        pass


class DuckDuckGoProvider(SearchProvider):
    """DuckDuckGo search provider - more reliable than Google scraping."""
    
    def search(self, query: str) -> List[SearchResult]:
        """Perform a DuckDuckGo HTML search and return result links."""
        try:
            url = f"https://duckduckgo.com/html/?q={quote(query)}"
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, 'html.parser')
            results = []

            # DuckDuckGo result selectors
            for a in soup.select('a.result__a'):
                href = a.get('href')
                title = a.get_text(strip=True)
                if href and href.startswith('http'):
                    # Clean up DuckDuckGo redirect URLs
                    clean_url = self._clean_duckduckgo_url(href)
                    if clean_url:
                        results.append(SearchResult(
                            url=clean_url,
                            title=title,
                            source="duckduckgo"
                        ))

            return results
            
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
            return []
    
    def _clean_duckduckgo_url(self, url: str) -> Optional[str]:
        """Clean DuckDuckGo redirect URLs to get the actual target URL."""
        try:
            # DuckDuckGo redirect pattern: /l/?uddg=<encoded_url>
            if '/l/?uddg=' in url:
                encoded_url = url.split('/l/?uddg=')[1]
                decoded = unquote(encoded_url)
                if decoded.startswith('http'):
                    return decoded
            
            # If no redirect pattern, return as-is
            return url if url.startswith('http') else None
            
        except Exception:
            return None


class SimpleScorer:
    """Simplified scoring system that focuses on what actually matters."""
    
    def __init__(self):
        # High-confidence publisher domains
        self.known_publishers = {
            'cephalofair.com': 100,          # Gloomhaven
            'fantasyflightgames.com': 100,   # Fantasy Flight
            'zmangames.com': 100,            # Z-Man Games
            'feuerland-spiele.de': 100,      # Ark Nova, Gaia Project
            'direwolf.com': 100,             # Dune: Imperium
            'fryxgames.se': 100,             # Terraforming Mars
            'aresgames.eu': 100,             # War of the Ring
            'greaterthangames.com': 100,     # Spirit Island
            'gmtgames.com': 100,             # Twilight Struggle
            'czechgames.com': 100,           # Through the Ages
            'stonemaiergames.com': 100,      # Scythe
            'roxley.com': 100,               # Brass series
            'asmodee.com': 100,              # Asmodee
            'days-of-wonder.com': 100,       # Days of Wonder
            'riograndegames.com': 100,       # Rio Grande
            'ravensburger.com': 100,         # Ravensburger (Alea imprint)
            'ravensburger.de': 100,
            'ravensburger.us': 100,
            'service.ravensburger.de': 100,
            'service.ravensburger.us': 100,
        }

        # Known reliable archives/aggregators for rulebooks (lower than official publishers)
        self.known_sources = {
            '1j1ju.com': 80,
        }
        
        # Sites to heavily penalize
        self.penalty_sites = [
            'reddit.com', 'forum', 'thread', 'review', 'discussion', 'wiki',
            'facebook.com', 'twitter.com', 'instagram.com', 'youtube.com',
            'boardgameatlas.com', 'amazon.com', 'ebay.com', 'etsy.com'
        ]
    
    def score(self, result: SearchResult, publisher: Optional[str] = None) -> int:
        """Score a search result. Higher scores are better."""
        score = 0
        url_l = result.url.lower()
        title_l = result.title.lower()
        domain = urlparse(result.url).netloc

        # Check for known publishers first (highest priority)
        for publisher_domain, points in self.known_publishers.items():
            if publisher_domain in domain:
                score += points
                break

        # Known sources boost (secondary to publishers)
        for src_domain, points in self.known_sources.items():
            if src_domain in domain:
                score += points
                break

        # Publisher name in domain or title
        if publisher:
            publisher_l = publisher.lower()
            if publisher_l in domain:
                score += 80
            if publisher_l in title_l:
                score += 60

        # Official-looking indicators
        if any(word in domain for word in ['official', 'company', 'publisher', 'games']):
            score += 30

        # Apply penalties for irrelevant sites
        for penalty in self.penalty_sites:
            if penalty in domain:
                score -= 50
                break

        return score


class TavilyProvider(SearchProvider):
    """Tavily search provider using the official API client.
    Returns results structured as SearchResult.
    """

    def __init__(self, timeout: int = 10, max_results: int = 8):
        super().__init__(timeout)
        self.max_results = max_results
        self._client = None

        if _TAVILY_AVAILABLE:
            api_key = os.environ.get("TAVILY_API_KEY")
            if api_key:
                try:
                    self._client = TavilyClient(api_key=api_key)
                except Exception as e:
                    logger.warning(f"Failed to initialize Tavily client: {e}")
            else:
                logger.debug("TAVILY_API_KEY not set; Tavily disabled")
        else:
            logger.debug("tavily-python package not available; Tavily disabled")

    def search(self, query: str) -> List[SearchResult]:
        results: List[SearchResult] = []
        if not self._client:
            return results
        try:
            # Use contextual search which tends to be best for LLMs
            resp = self._client.search(
                query,
                max_results=self.max_results,
                include_answer=False,
            )
            # resp format example: {"results": [{"url": ..., "title": ... , "content": ...}, ...]}
            for item in resp.get("results", [])[: self.max_results]:
                url = item.get("url") or ""
                title = item.get("title") or item.get("url") or ""
                if url and url.startswith("http"):
                    results.append(
                        SearchResult(url=url, title=title, source="tavily")
                    )
        except Exception as e:
            logger.warning(f"Tavily search failed: {e}")
        return results


class WebSearchFallback:
    """Simplified web search fallback that actually works."""
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.scorer = SimpleScorer()
        # Compose: use agentic Tavily loop if available; else direct provider; else DDG
        self._tavily_agent = None
        try:
            from .agent_tavily import TavilyAgent  # lazy import to reduce deps at import time
            # Initialize agent only if key available
            if os.environ.get("TAVILY_API_KEY"):
                self._tavily_agent = TavilyAgent()
        except Exception:
            self._tavily_agent = None
        # Fallback direct providers
        tavily = TavilyProvider(timeout)
        self.provider = tavily if getattr(tavily, "_client", None) else DuckDuckGoProvider(timeout)
    
    def search_official_website(self, game_name: str, publisher: Optional[str] = None, max_results: int = 5) -> List[SearchResult]:
        """
        Search for the game's official website.
        Simplified approach that focuses on effectiveness.
        """
        # Simple, effective queries
        queries = [
            f'"{game_name}" board game official website',
            f'"{game_name}" boardgame official website',
        ]
        
        # Add publisher-specific queries if available
        if publisher:
            queries.insert(0, f'"{publisher}" {game_name} official website')
            queries.insert(1, f'"{publisher}" board games official website')

        all_candidates = []
        seen_urls = set()

        # Try agentic Tavily first (if available)
        if self._tavily_agent:
            try:
                agent_results = self._tavily_agent.search_official_website(game_name, publisher, max_results)
                for r in agent_results:
                    r.score = self.scorer.score(r, publisher)
                agent_results = [r for r in agent_results if r.score > 20]
                agent_results.sort(key=lambda x: x.score, reverse=True)
                if agent_results:
                    return agent_results[:max_results]
            except Exception as e:
                logger.debug(f"Agentic Tavily official website search failed: {e}")

        def contains_game_tokens(result: SearchResult, game: str) -> bool:
            tokens = [t.lower() for t in game.split() if len(t) > 3]
            title_l = result.title.lower()
            url_l = result.url.lower()
            hits = sum(1 for t in tokens if (t in title_l or t in url_l))
            return hits >= max(1, len(tokens) // 2)

        for query in queries:
            try:
                results = self.provider.search(query)
                for result in results:
                    if result.url not in seen_urls:
                        seen_urls.add(result.url)
                        
                        # Score this candidate
                        result.score = self.scorer.score(result, publisher)
                        # Boost if the title/url contains game tokens to avoid publisher-only or unrelated hits
                        try:
                            if contains_game_tokens(result, game_name):
                                result.score += 40
                        except Exception:
                            pass
                        
                        # Only include promising candidates
                        if result.score > 20:
                            all_candidates.append(result)
                
                # Small delay between queries
                time.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")
                continue

        # Sort by score and return top candidates
        all_candidates.sort(key=lambda x: x.score, reverse=True)
        return all_candidates[:max_results]

    def search_official_rulebook(self, game_name: str, publisher: Optional[str] = None, max_results: int = 5) -> List[SearchResult]:
        """
        Search for official rulebook PDFs.
        Simplified approach focused on finding downloadable rulebooks.
        """
        # Simple, effective queries for rulebooks
        queries = [
            f'"{game_name}" board game rulebook filetype:pdf',
            f'"{game_name}" board game rules pdf',
            f'"{game_name}" official rulebook download',
            # Prefer publisher domains
            f'site:ravensburger.* "{game_name}" filetype:pdf',
            f'site:service.ravensburger.* "{game_name}" filetype:pdf',
            f'site:alea.* "{game_name}" filetype:pdf',
        ]
        
        # Add publisher-specific queries if available
        if publisher:
            queries.insert(0, f'"{publisher}" {game_name} rulebook filetype:pdf')
            queries.insert(1, f'"{publisher}" {game_name} official rules pdf')

        def contains_game_tokens(result: SearchResult, game: str) -> bool:
            tokens = [t.lower() for t in game.split() if len(t) > 3]
            title_l = result.title.lower()
            url_l = result.url.lower()
            hits = sum(1 for t in tokens if (t in title_l or t in url_l))
            return hits >= max(1, len(tokens) // 2)

        all_results = []
        seen_urls = set()

        # Try agentic Tavily first (if available)
        if self._tavily_agent:
            try:
                agent_results = self._tavily_agent.search_rulebook(game_name, publisher, max_results)
                # prioritize PDFs
                pdf_results = [r for r in agent_results if '.pdf' in r.url.lower()]
                others = [r for r in agent_results if '.pdf' not in r.url.lower()]
                ordered = pdf_results + others
                if ordered:
                    return ordered[:max_results]
            except Exception as e:
                logger.debug(f"Agentic Tavily rulebook search failed: {e}")

        for query in queries:
            try:
                results = self.provider.search(query)
                
                for result in results:
                    if result.url in seen_urls:
                        continue
                    # Filter out clearly unrelated PDFs that do not include game tokens
                    if not contains_game_tokens(result, game_name):
                        # Still allow if domain is a known publisher (handled by scorer)
                        domain = urlparse(result.url).netloc
                        if not any(pub in domain for pub in self.scorer.known_publishers.keys()):
                            continue
                    seen_urls.add(result.url)
                    all_results.append(result)
                
                # Small delay between queries
                time.sleep(0.5)
                
                # If we got good results, we can stop early
                if len(all_results) >= max_results:
                    break
                    
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")
                continue

        # Return top results, prioritizing PDFs
        pdf_results = [r for r in all_results if '.pdf' in r.url.lower()]
        other_results = [r for r in all_results if '.pdf' not in r.url.lower()]

        # Sort PDFs by publisher preference first
        def pdf_rank(r: SearchResult) -> int:
            try:
                domain = urlparse(r.url).netloc
                if any(pub in domain for pub in self.scorer.known_publishers.keys()):
                    return 0
            except Exception:
                pass
            return 1

        pdf_results.sort(key=pdf_rank)
        ordered = pdf_results + other_results
        return ordered[:max_results]


