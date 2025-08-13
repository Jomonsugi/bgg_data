"""
Web search system using Tavily (primary) with DuckDuckGo (fallback).
Keeps the existing scoring and interface so downstream code remains compatible.
"""

import logging
import time
from typing import List, Dict, Optional, Tuple
from urllib.parse import quote, urlparse, unquote
from pathlib import Path
import json
import time
import math
import random
import base64
from dataclasses import dataclass
from abc import ABC, abstractmethod

import os
import requests
from bs4 import BeautifulSoup
from ...config import PROJECT_ROOT

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
        # Basic retry policy for robustness
        try:
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            retries = Retry(
                total=3,
                backoff_factor=0.3,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "OPTIONS"],
            )
            adapter = HTTPAdapter(max_retries=retries)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
        except Exception:
            pass
    
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
        # Load configurable priors instead of hardcoding
        self._store_dir = PROJECT_ROOT / "bgg_data_cache"
        self._store_dir.mkdir(parents=True, exist_ok=True)
        self._config_path = self._store_dir / "search_config.json"
        cfg = self._load_config()
        # High-confidence publisher domains (user/data driven)
        self.known_publishers: Dict[str, int] = cfg.get('known_publishers', {})
        # Known reliable archives/aggregators for rulebooks (lower than official publishers)
        self.known_sources: Dict[str, int] = cfg.get('known_sources', {})
        # Sites to heavily penalize
        self.penalty_sites: List[str] = cfg.get('penalty_sites', [])
        # Dynamic domain history (learn over time)
        self._store_path = self._store_dir / "search_scores.json"
        self._history: Dict[str, Dict] = self._load_history()
        # Dynamic publisher → domains mapping that grows over time
        self._pub_store_path = self._store_dir / "search_publishers.json"
        self._dynamic_publishers: Dict[str, Dict[str, Dict[str, float]]] = self._load_publishers()
        self._decay_half_life_days: float = 30.0
        self._ucb_c: float = 40.0
        self._epsilon_base: float = 0.15
    
    def score(self, result: SearchResult, publisher: Optional[str] = None) -> int:
        """Score a search result. Higher scores are better."""
        score = 0
        url_l = result.url.lower()
        title_l = result.title.lower()
        domain = urlparse(result.url).netloc
        path = urlparse(result.url).path.lower()
        path = urlparse(result.url).path.lower()

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

            # Dynamic publisher-domain boosts learned over time
            try:
                pub_map = self._dynamic_publishers.get(publisher_l, {})
                # Exact netloc match preferred; fallback to endswith
                if domain in pub_map:
                    score += int(pub_map[domain].get('points', 100))
                else:
                    for d, meta in pub_map.items():
                        if domain.endswith(d):
                            score += int(meta.get('points', 80))
                            break
            except Exception:
                pass

        # Official-looking indicators
        if any(word in domain for word in ['official', 'company', 'publisher', 'games']):
            score += 30

        # Apply penalties for irrelevant sites
        for penalty in self.penalty_sites:
            if penalty in domain:
                score -= 50
                break

        # Modest nudge for BGG threads that look like rulebook posts (still below publisher boost)
        try:
            if 'boardgamegeek.com' in domain and '/thread/' in path:
                if 'rulebook' in title_l or 'rulebook' in url_l:
                    score += 25
        except Exception:
            pass

        # Prefer direct PDFs
        if path.endswith('.pdf'):
            score += 30

        # Dynamic adjustment based on historic success/failure for this domain (with decay)
        try:
            now_ts = self._now()
            succ, fail, visits = self._get_domain_stats(domain, publisher, now_ts)
            total = succ + fail
            if total >= 1:
                success_rate = succ / max(1.0, total)
                # Map success_rate in [0,1] to roughly [-60, +120]
                dynamic = int((success_rate - 0.5) * 360)
                if dynamic > 0 and any(pub in domain for pub in self.known_publishers.keys()):
                    dynamic = min(dynamic, 60)
                score += max(-80, min(120, dynamic))
            # UCB exploration bonus (encourage exploring low-visit domains)
            total_n = int(self._history.get('_total_n', 0))
            ucb = self._ucb_c * math.sqrt(max(0.0, math.log(max(1, total_n)) / max(1, visits)))
            score += int(ucb)
            # Epsilon-greedy random boost, stronger if few visits
            epsilon = self._epsilon_base if visits < 3 else (self._epsilon_base * 0.3)
            if random.random() < epsilon:
                score += int(random.uniform(5, 25 if visits < 3 else 10))
        except Exception:
            pass

        return score

    def record_success(self, url: str, publisher: Optional[str] = None) -> None:
        try:
            domain = urlparse(url).netloc
            if not domain:
                return
            now_ts = self._now()
            entry = self._history.setdefault(domain, {"success": 0.0, "failure": 0.0, "n": 0.0, "last_ts": now_ts, "by_publisher": {}})
            self._apply_decay(entry, now_ts)
            entry["success"] = float(entry.get("success", 0.0)) + 1.0
            entry["n"] = float(entry.get("n", 0.0)) + 1.0
            entry["last_ts"] = now_ts
            if publisher:
                pub = entry["by_publisher"].setdefault(publisher.lower(), {"success": 0.0, "failure": 0.0, "n": 0.0, "last_ts": now_ts})
                self._apply_decay(pub, now_ts)
                pub["success"] = float(pub.get("success", 0.0)) + 1.0
                pub["n"] = float(pub.get("n", 0.0)) + 1.0
                pub["last_ts"] = now_ts
                # Promote this domain into dynamic publisher map with growing points
                try:
                    pl = publisher.lower()
                    pmap = self._dynamic_publishers.setdefault(pl, {})
                    meta = pmap.setdefault(domain, {"points": 100.0, "success": 0.0})
                    meta["success"] = float(meta.get("success", 0.0)) + 1.0
                    # Increase points gradually up to 140
                    meta["points"] = float(min(140.0, meta.get("points", 100.0) + 5.0))
                    self._save_publishers()
                except Exception:
                    pass
            self._history['_total_n'] = int(self._history.get('_total_n', 0)) + 1
            self._save_history()
        except Exception:
            pass

    def record_failure(self, url: str, publisher: Optional[str] = None) -> None:
        try:
            domain = urlparse(url).netloc
            if not domain:
                return
            now_ts = self._now()
            entry = self._history.setdefault(domain, {"success": 0.0, "failure": 0.0, "n": 0.0, "last_ts": now_ts, "by_publisher": {}})
            self._apply_decay(entry, now_ts)
            entry["failure"] = float(entry.get("failure", 0.0)) + 1.0
            entry["n"] = float(entry.get("n", 0.0)) + 1.0
            entry["last_ts"] = now_ts
            if publisher:
                pub = entry["by_publisher"].setdefault(publisher.lower(), {"success": 0.0, "failure": 0.0, "n": 0.0, "last_ts": now_ts})
                self._apply_decay(pub, now_ts)
                pub["failure"] = float(pub.get("failure", 0.0)) + 1.0
                pub["n"] = float(pub.get("n", 0.0)) + 1.0
                pub["last_ts"] = now_ts
                # Slightly reduce dynamic publisher-domain points on failure (not below 60)
                try:
                    pl = publisher.lower()
                    pmap = self._dynamic_publishers.setdefault(pl, {})
                    meta = pmap.setdefault(domain, {"points": 90.0, "success": 0.0})
                    meta["points"] = float(max(60.0, meta.get("points", 90.0) - 3.0))
                    self._save_publishers()
                except Exception:
                    pass
            self._history['_total_n'] = int(self._history.get('_total_n', 0)) + 1
            self._save_history()
        except Exception:
            pass

    def _load_history(self) -> Dict[str, Dict]:
        try:
            if self._store_path.exists():
                with open(self._store_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {}

    def _save_history(self) -> None:
        try:
            with open(self._store_path, 'w', encoding='utf-8') as f:
                json.dump(self._history, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _load_config(self) -> Dict:
        # Initialize with reasonable defaults and merge persisted config
        default = {
            "known_publishers": {},
            "known_sources": {"1j1ju.com": 80},
            "penalty_sites": [
                "reddit.com", "forum", "thread", "review", "discussion", "wiki",
                "facebook.com", "twitter.com", "instagram.com", "youtube.com",
                "boardgameatlas.com", "amazon.com", "ebay.com", "etsy.com"
            ],
        }
        try:
            if self._config_path.exists():
                with open(self._config_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                if isinstance(cfg, dict):
                    # shallow merge
                    for k, v in cfg.items():
                        if isinstance(v, dict):
                            default[k].update(v)
                        else:
                            default[k] = v
        except Exception:
            pass
        return default

    def _load_publishers(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        try:
            if self._pub_store_path.exists():
                with open(self._pub_store_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {}

    def _save_publishers(self) -> None:
        try:
            with open(self._pub_store_path, 'w', encoding='utf-8') as f:
                json.dump(self._dynamic_publishers, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _now(self) -> float:
        return float(time.time())

    def _apply_decay(self, entry: Dict, now_ts: float) -> None:
        try:
            last = float(entry.get('last_ts', now_ts))
            if last <= 0:
                entry['last_ts'] = now_ts
                return
            dt_days = max(0.0, (now_ts - last) / 86400.0)
            if dt_days <= 0:
                return
            # exponential decay: factor = 0.5 ** (dt / half_life)
            factor = math.pow(0.5, dt_days / max(1e-6, self._decay_half_life_days))
            for key in ('success', 'failure', 'n'):
                entry[key] = float(entry.get(key, 0.0)) * factor
            entry['last_ts'] = now_ts
        except Exception:
            pass

    def _get_domain_stats(self, domain: str, publisher: Optional[str], now_ts: float) -> Tuple[float, float, float]:
        entry = self._history.get(domain, {})
        if not entry:
            return 0.0, 0.0, 0.0
        self._apply_decay(entry, now_ts)
        succ = float(entry.get('success', 0.0))
        fail = float(entry.get('failure', 0.0))
        visits = float(entry.get('n', succ + fail))
        if publisher:
            try:
                pub = entry.get('by_publisher', {}).get(publisher.lower())
                if pub:
                    self._apply_decay(pub, now_ts)
                    # blend domain + publisher stats (weighted 70/30 to publisher if present)
                    ps = float(pub.get('success', 0.0))
                    pf = float(pub.get('failure', 0.0))
                    pv = float(pub.get('n', ps + pf))
                    succ = 0.7 * ps + 0.3 * succ
                    fail = 0.7 * pf + 0.3 * fail
                    visits = max(visits, pv)
            except Exception:
                pass
        return succ, fail, max(1.0, visits)


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

        # Try Tavily first (if available)
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
                logger.debug(f"Tavily official website search failed: {e}")

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

    def search_official_rulebook(self, game_name: str, publisher: Optional[str] = None, max_results: int = 5, relaxed: bool = False) -> List[SearchResult]:
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
            # Last-resort BGG thread query for community-posted rulebooks (kept after publisher priorities)
            f'"{game_name}" rulebook site:boardgamegeek.com/thread',
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

        # In relaxed mode (after official site failed), prioritize high-scoring trusted domains dynamically,
        # while still giving a small nudge to BGG threads that look like rulebook posts
        if relaxed:
            def other_sort_key(r: SearchResult):
                try:
                    u = urlparse(r.url)
                    is_bgg_thread = ("boardgamegeek.com" in u.netloc and "/thread/" in u.path)
                except Exception:
                    is_bgg_thread = False
                score_dyn = 0
                try:
                    score_dyn = self.scorer.score(r, publisher)
                except Exception:
                    score_dyn = 0
                # Sort by (bgg_thread_priority, negative dynamic score) so lower tuple wins
                return (0 if is_bgg_thread else 1, -score_dyn)

            other_results.sort(key=other_sort_key)

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
        return ordered[: (max_results if not relaxed else max(max_results, 8))]


