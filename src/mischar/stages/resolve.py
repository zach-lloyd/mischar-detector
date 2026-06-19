"""
Stage 2: Citation resolution via CourtListener.

Resolves a ``ParsedCitation`` to its full opinion text by querying the
CourtListener API. This stage handles rate limiting, retries, and
caching to avoid redundant API calls.

Two distinct failure modes exist:
- **case-not-found**: CourtListener has no record matching the citation.
- **text-not-retrieved**: The case exists but its opinion text isn't available
  (some cases only have metadata, not full text).

Both produce ``Abstention`` objects, not exceptions.
"""

from __future__ import annotations

import time
from datetime import date

import httpx

from mischar.cache import Cache
from mischar.logging import get_logger
from mischar.models.client import retry_with_backoff
from mischar.types import Abstention, ParsedCitation, ResolvedCase

log = get_logger("resolve")


# ---------------------------------------------------------------------------
# CourtListener API client
# ---------------------------------------------------------------------------


class CourtListenerAPIError(Exception):
    """
    Infrastructure error from CourtListener (5xx, network failure, etc.).

    Distinct from a "not found" result, which is a normal pipeline outcome.
    """

    pass


class CourtListenerClient:
    """
    HTTP client for the CourtListener REST API.

    Handles authentication, rate limiting (token-bucket style), and
    retries with exponential backoff on transient errors (429, 5xx).

    Args:
        api_key: CourtListener API key for authentication.
        base_url: API base URL. Defaults to CourtListener's production API.
        rate_limit_per_minute: Maximum requests per minute. CourtListener's
            free tier allows 60/minute.
        max_retries: Number of retry attempts on transient failures.
        timeout_seconds: HTTP request timeout.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://www.courtlistener.com/api/rest/v4/",
        rate_limit_per_minute: int = 60,
        max_retries: int = 5,
        timeout_seconds: int = 30,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries

        # HTTP client with auth header. CourtListener uses token-based auth.
        self._http = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds),
            headers={"Authorization": f"Token {api_key}"},
        )

        # Simple rate limiter: track timestamps of recent requests and
        # sleep if we're about to exceed the per-minute limit.
        self._rate_limit = rate_limit_per_minute
        self._request_timestamps: list[float] = []

        log.info(
            "courtlistener_client_initialized",
            base_url=self._base_url,
            rate_limit=rate_limit_per_minute,
        )


    def lookup_citation(self, citation: ParsedCitation) -> dict | None:
        """
        Resolve a citation to a CourtListener cluster via the dedicated
        Citation Lookup API (``/citation-lookup/``).

        Unlike a relevance-ranked search, this endpoint performs an
        *exact* citation match using Eyecite, so it returns the specific
        case for the given volume/reporter/page rather than the most
        popular hit. This avoids attaching the wrong opinion to a claim.

        Args:
            citation: The parsed citation to look up.

        Returns:
            The matched CourtListener cluster object as a dict (which
            includes ``id``, ``case_name``, ``date_filed``, etc.), or
            None if the citation was not found, invalid, or ambiguous.

        Raises:
            CourtListenerAPIError: On infrastructure failure after retries.
        """
        # The citation-lookup endpoint takes the volume/reporter/page
        # triad directly and normalizes the reporter internally.
        payload = {
            "volume": citation.volume,
            "reporter": citation.reporter,
            "page": citation.page,
        }
        cite_str = f"{citation.volume} {citation.reporter} {citation.page}"

        log.debug(
            "courtlistener_lookup",
            cite=cite_str,
            case_name=citation.case_name,
        )

        # Returns a list with one entry per parsed citation. We send a
        # single citation, so we expect at most one entry.
        results = self._post("citation-lookup/", data=payload)

        if not results:
            log.info("courtlistener_not_found", cite=cite_str)

            return None

        entry = results[0]
        status = entry.get("status")
        clusters = entry.get("clusters", [])

        # status 200 = found and looked up exactly once. Anything else
        # (404 not found, 400 invalid reporter, 300 multiple/ambiguous
        # matches) we treat as unresolved and drop, rather than risk
        # attaching the wrong case to a claim.
        if status != 200 or not clusters:
            log.info(
                "courtlistener_not_resolved",
                cite=entry.get("citation", cite_str),
                status=status,
                n_clusters=len(clusters),
                error=entry.get("error_message", ""),
            )

            return None

        # Exact match — return the single matched cluster.
        return clusters[0]


    def fetch_opinion_text(self, cluster: dict) -> str | None:
        """
        Fetch the full opinion text for a resolved cluster.

        CourtListener organizes opinions under "clusters" (a case can have
        multiple opinions — majority, dissent, concurrence). We fetch all
        opinions in the cluster and concatenate them, since the pipeline
        needs to search across the full text.

        The cluster object returned by the citation-lookup endpoint already
        embeds the ``sub_opinions`` list, so we read it directly and skip a
        redundant ``clusters/{id}/`` round-trip. If the passed object lacks
        that list (e.g. a differently-shaped cluster), we fall back to
        fetching the cluster record once.

        Args:
            cluster: The CourtListener cluster object (as returned by
                ``lookup_citation``), containing ``id`` and ``sub_opinions``.

        Returns:
            The full opinion text as a string, or None if no text is
            available for this case.

        Raises:
            CourtListenerAPIError: On infrastructure failure after retries.
        """
        cluster_id = cluster.get("id")
        log.debug("courtlistener_fetch_opinion", cluster_id=cluster_id)

        # The citation-lookup cluster object already carries the sub-opinion
        # URLs, so no extra request is needed in the common case.
        sub_opinion_urls = cluster.get("sub_opinions", [])

        # Fallback: if the embedded object didn't include sub_opinions, fetch
        # the cluster record directly (the pre-optimization behavior).
        if not sub_opinion_urls and cluster_id is not None:
            cluster_data = self._get(f"clusters/{cluster_id}/")
            sub_opinion_urls = cluster_data.get("sub_opinions", [])

        if not sub_opinion_urls:
            log.info("courtlistener_no_opinions", cluster_id=cluster_id)

            return None

        # Fetch each sub-opinion and collect the text. CourtListener
        # stores opinion text in several fields; we prefer plain_text,
        # then html_with_citations, then html.
        texts = []
        for url in sub_opinion_urls:
            # sub_opinion_urls are full URLs; extract the ID.
            opinion_id = url.rstrip("/").split("/")[-1]
            opinion_data = self._get(f"opinions/{opinion_id}/")

            text = (
                opinion_data.get("plain_text")
                or opinion_data.get("html_with_citations")
                or opinion_data.get("html")
                or ""
            )

            # If we got HTML, do a basic strip of tags. This is a rough
            # conversion — good enough for chunking and embedding, not
            # for display.
            if text and "<" in text:
                text = _strip_html_tags(text)

            if text.strip():
                texts.append(text.strip())

        if not texts:
            log.info("courtlistener_no_text", cluster_id=cluster_id)

            return None

        # Join multiple opinions (majority + dissent, etc.) with a
        # separator so chunking can treat them as one document.
        full_text = "\n\n---\n\n".join(texts)
        log.info(
            "courtlistener_text_fetched",
            cluster_id=cluster_id,
            opinions=len(texts),
            chars=len(full_text),
        )

        return full_text


    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()

    # ------------------------------------------------------------------
    # Internal HTTP helper
    # ------------------------------------------------------------------

    def _get(self, endpoint: str, params: dict | None = None) -> dict:
        """
        Make a rate-limited, retried GET request to CourtListener.

        Handles:
        - Rate limiting: sleeps if we're approaching the per-minute limit.
        - Retries with backoff on 429 (rate limited) and 5xx (server error).
        - Raises CourtListenerAPIError on non-retryable failures.

        Args:
            endpoint: The CourtListener API endpoint to try.
            params: Optional dictionary of parameters to send with the HTTP GET 
                request.
        
        Returns:
            The CourtListener response.
        """
        self._enforce_rate_limit()
        url = f"{self._base_url}/{endpoint.lstrip('/')}"


        def _call() -> httpx.Response:
            response = self._http.get(url, params=params)

            # 429 = rate limited, 5xx = server error. Both are retryable.
            if response.status_code == 429:
                raise CourtListenerAPIError(
                    f"Rate limited (429) on {endpoint}"
                )
            if response.status_code >= 500:
                raise CourtListenerAPIError(
                    f"Server error ({response.status_code}) on {endpoint}"
                )

            # 404 is a valid "not found" — don't retry, just return an
            # empty result that the caller handles.
            if response.status_code == 404:
                return response

            # Any other non-2xx is unexpected.
            response.raise_for_status()

            return response

        try:
            response = retry_with_backoff(
                _call,
                max_retries=self._max_retries,
                retryable_exceptions=(CourtListenerAPIError, httpx.HTTPError),
                context=f"courtlistener {endpoint}",
            )
        except (CourtListenerAPIError, httpx.HTTPError) as exc:
            raise CourtListenerAPIError(
                f"CourtListener API request failed after retries: {exc}"
            ) from exc

        # 404 returns an empty dict (caller checks for missing data).
        if response.status_code == 404:
            return {}

        return response.json()


    def _post(self, endpoint: str, data: dict | None = None) -> list | dict:
        """
        Make a rate-limited, retried POST request to CourtListener.

        Same retry and rate-limit semantics as :meth:`_get`, but sends
        form data via POST. Used for the citation-lookup endpoint, which
        returns a JSON list (one entry per parsed citation).

        Args:
            endpoint: The CourtListener API endpoint to call.
            data: Form fields to send in the POST body.

        Returns:
            The parsed JSON response (a list for citation-lookup), or an
            empty list on a 404.
        """
        self._enforce_rate_limit()
        url = f"{self._base_url}/{endpoint.lstrip('/')}"

        def _call() -> httpx.Response:
            response = self._http.post(url, data=data)

            # 429 = rate limited, 5xx = server error. Both are retryable.
            if response.status_code == 429:
                raise CourtListenerAPIError(
                    f"Rate limited (429) on {endpoint}"
                )
            if response.status_code >= 500:
                raise CourtListenerAPIError(
                    f"Server error ({response.status_code}) on {endpoint}"
                )

            if response.status_code == 404:
                return response

            # Any other non-2xx is unexpected.
            response.raise_for_status()

            return response

        try:
            response = retry_with_backoff(
                _call,
                max_retries=self._max_retries,
                retryable_exceptions=(CourtListenerAPIError, httpx.HTTPError),
                context=f"courtlistener {endpoint}",
            )
        except (CourtListenerAPIError, httpx.HTTPError) as exc:
            raise CourtListenerAPIError(
                f"CourtListener API request failed after retries: {exc}"
            ) from exc

        if response.status_code == 404:
            return []

        return response.json()


    def _enforce_rate_limit(self) -> None:
        """
        Sleep if needed to stay within the per-minute rate limit.

        Uses a sliding window: we track the timestamp of each request
        and sleep if the oldest request in our window is less than 60
        seconds ago and we've hit the limit.
        """
        now = time.time()

        # Discard timestamps older than 60 seconds.
        self._request_timestamps = [
            ts for ts in self._request_timestamps if now - ts < 60
        ]

        # If we've used all our budget, sleep until the oldest request
        # ages out of the window.
        if len(self._request_timestamps) >= self._rate_limit:
            sleep_time = 60 - (now - self._request_timestamps[0])
            if sleep_time > 0:
                log.info("courtlistener_rate_limit_sleep", seconds=round(sleep_time, 1))
                time.sleep(sleep_time)

        # Record this request.
        self._request_timestamps.append(time.time())


# ---------------------------------------------------------------------------
# Stage function
# ---------------------------------------------------------------------------


def resolve_citation(
    citation: ParsedCitation,
    client: CourtListenerClient,
    cache: Cache,
) -> ResolvedCase | Abstention:
    """
    Resolve a parsed citation to its full opinion text.

    Checks the cache first. On miss, queries CourtListener to find the
    case and fetch its opinion text.

    Args:
        citation: The parsed citation to resolve.
        client: CourtListener API client.
        cache: Pipeline cache instance.

    Returns:
        A ``ResolvedCase`` with full text on success, or an ``Abstention``
        with reason ``case-not-found`` or ``text-not-retrieved`` on failure.
    """
    # Build a cache key from the citation's identifying components.
    # Volume + reporter + page uniquely identifies a case citation.
    cache_key = Cache.make_key(
        citation.volume,
        citation.reporter,
        citation.page,
    )

    # Check cache first — avoids redundant API calls for citations
    # we've already resolved.
    cached = cache.get("resolve", cache_key)
    if cached is not None:
        log.debug("resolve_cache_hit", cite=citation.raw_text)

        return cached

    # Cache miss — query CourtListener.
    log.info("resolve_lookup", cite=citation.raw_text)

    # Step 1: Find the case in CourtListener.
    case_record = client.lookup_citation(citation)
    if case_record is None:
        result = Abstention(
            reason="case-not-found",
            details=f"No CourtListener match for {citation.raw_text}",
        )
        # Cache the not-found result too — no point re-querying.
        cache.set("resolve", cache_key, result)

        return result

    # Extract the cluster ID (CourtListener's grouping for a case's opinions).
    cluster_id = str(case_record.get("cluster_id") or case_record.get("id", ""))
    case_name = case_record.get("caseName") or case_record.get("case_name") or "Unknown"
    court = case_record.get("court") or ""

    # Parse the decision date if available.
    decided_at = _parse_date(case_record.get("dateFiled") or case_record.get("date_filed"))

    # Step 2: Fetch the full opinion text. Pass the cluster object we
    # already have so fetch_opinion_text can reuse its embedded
    # sub_opinions list instead of re-fetching the cluster.
    full_text = client.fetch_opinion_text(case_record)
    if not full_text:
        result = Abstention(
            reason="text-not-retrieved",
            details=f"Case found ({case_name}) but no opinion text available",
        )
        cache.set("resolve", cache_key, result)

        return result

    # Success — build the ResolvedCase and cache it.
    resolved = ResolvedCase(
        courtlistener_id=cluster_id,
        case_name=case_name,
        citation_string=citation.raw_text,
        full_text=full_text,
        decided_at=decided_at,
        court=court,
    )

    cache.set("resolve", cache_key, resolved)
    log.info(
        "resolve_success",
        cite=citation.raw_text,
        case_name=case_name,
        text_chars=len(full_text),
    )

    return resolved


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_date(date_str: str | None) -> date | None:
    """
    Parse a date string from CourtListener (format: YYYY-MM-DD).

    Args:
        date_str: The date string to be parsed.
    
    Returns:
        The date in YYYY-MM-DD format.
    """
    if not date_str:
        return None
    try:
        return date.fromisoformat(date_str)
    except (ValueError, TypeError):
        return None


def _strip_html_tags(html: str) -> str:
    """
    Quick-and-dirty HTML tag removal.

    This is intentionally basic — we only need clean-enough text for
    chunking and embedding, not a perfect HTML-to-text conversion.
    A dedicated library (like beautifulsoup) would be overkill here.

    Args:
        html: The HTML to parse and convert to text.
    
    Returns:
        The HTML converted to text and cleaned sufficiently for chunking and 
        embedding.
    """
    import re

    # Remove HTML tags.
    text = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()
