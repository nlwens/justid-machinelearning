"""A small, well-behaved client for the Rechtspraak Open Data API.

This module owns every outbound HTTP call in the project. Concentrating them
here means the rate limit, the retry policy and the caching rule are enforced
once, instead of being re-implemented (slightly differently) in each script.

The API has two halves, mirrored by the two main functions below:

    search_eclis()    query the ECLI index -> a list of ECLI identifiers
    fetch_document()  fetch one ruling by ECLI -> its XML (metadata + text)

Design notes
------------
Rate limiting is applied process-wide rather than per-call, because the
crawler issues tens of thousands of requests in a tight loop and a per-call
`sleep` would not actually bound the rate.

Responses from the content endpoint are cached on disk. A full crawl takes
hours; being able to interrupt it and resume without re-downloading is the
difference between a pipeline you can iterate on and one you run once and
never dare touch again.
"""

from __future__ import annotations

import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

import requests

import _paths  # noqa: F401
import config

# The Atom namespace used by the index endpoint's feed.
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


class RateLimiter:
    """Enforces a minimum interval between successive requests, process-wide.

    Rechtspraak allows 10 requests/second; `config.REQUESTS_PER_SECOND` sets
    what we actually use. This is a spacing limiter rather than a token
    bucket: we never want to burst, so there is no allowance to accumulate.

    Thread safety is not optional here. The download step runs several worker
    threads, and without the lock two of them can read the same
    `_last_call_at`, both conclude they may proceed, and fire simultaneously
    -- which is exactly the burst the limiter exists to prevent. The lock is
    held across the sleep so that waiting threads queue up and are spaced out
    one after another.
    """

    def __init__(self, requests_per_second: float) -> None:
        self._min_interval = 1.0 / requests_per_second
        self._last_call_at = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Block until enough time has passed since the previous request."""
        with self._lock:
            elapsed = time.monotonic() - self._last_call_at
            remaining = self._min_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
            self._last_call_at = time.monotonic()


# One limiter and one connection pool shared by the whole process.
_limiter = RateLimiter(config.REQUESTS_PER_SECOND)
_session = requests.Session()
_session.headers.update({"User-Agent": config.USER_AGENT})

# The default urllib3 pool holds 10 connections. Sizing it to the worker
# count keeps threads from discarding and reopening TLS connections, which
# would otherwise show up as a steady stream of "connection pool is full"
# warnings and noticeably slower downloads.
_adapter = requests.adapters.HTTPAdapter(
    pool_connections=config.DOWNLOAD_WORKERS,
    pool_maxsize=config.DOWNLOAD_WORKERS,
)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)


def _get(url: str, params: dict | None = None) -> str:
    """Perform a rate-limited GET with retries, returning the response body.

    Retries on connection errors and on the status codes that indicate a
    transient problem (429 too many requests, 5xx server errors). A 404 or
    other client error is returned to the caller as an exception immediately,
    because retrying it would just waste the request budget.

    :raises requests.HTTPError: if the request still fails after MAX_RETRIES.
    """
    last_error: Exception | None = None

    for attempt in range(config.MAX_RETRIES):
        _limiter.wait()
        try:
            response = _session.get(
                url, params=params, timeout=config.REQUEST_TIMEOUT_SECONDS
            )
            # Transient server-side conditions: back off and try again.
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.HTTPError(
                    f"transient status {response.status_code}", response=response
                )
            response.raise_for_status()
            return response.text

        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            # A non-transient client error (e.g. 404) should not be retried.
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                raise

            last_error = exc
            backoff = config.RETRY_BACKOFF_SECONDS * (2**attempt)
            print(
                f"  request failed ({exc}); retry {attempt + 1}/{config.MAX_RETRIES} "
                f"in {backoff:.0f}s"
            )
            time.sleep(backoff)

    raise requests.HTTPError(
        f"giving up on {url} after {config.MAX_RETRIES} attempts"
    ) from last_error


def fetch_taxonomy(name: str) -> str:
    """Download one controlled vocabulary (e.g. 'rechtsgebieden') as XML text."""
    return _get(config.TAXONOMY_URLS[name])


def search_eclis(
    date_from: str,
    date_to: str,
    subject_uri: str | None = None,
    limit: int | None = None,
) -> list[str]:
    """Return the ECLIs of rulings published within a date range.

    The index is walked page by page using the `from` offset, because the
    endpoint caps a single response at `config.MAX_RESULTS_PER_PAGE` entries.

    :param date_from: inclusive start date, ``YYYY-MM-DD``.
    :param date_to: inclusive end date, ``YYYY-MM-DD``.
    :param subject_uri: optional PSI law-area URI. Filtering on a top-level
        area also returns rulings tagged with its sub-areas, which is what
        makes it usable as a stratification key.
    :param limit: stop once this many ECLIs have been collected. ``None``
        collects every match.
    :returns: ECLI strings, in the order the API returned them.

    Note that date ranges are *stable*: the set of rulings published in a past
    year does not change, so repeating this call later reproduces the same
    result. This is the property the 2024 offset-based crawler lacked.
    """
    collected: list[str] = []
    offset = 0

    while True:
        # `date` is deliberately passed twice -- the API reads a pair of date
        # parameters as a range. requests encodes a list as repeated keys.
        params: list[tuple[str, str]] = [
            ("date", date_from),
            ("date", date_to),
            ("max", str(config.MAX_RESULTS_PER_PAGE)),
            ("from", str(offset)),
        ]
        if config.REQUIRE_FULL_DOCUMENT:
            # Restrict to ECLIs that actually carry a full-text document.
            # Around half of all ECLIs are metadata-only registrations.
            params.append(("return", "DOC"))
        if subject_uri is not None:
            params.append(("subject", subject_uri))

        feed = ET.fromstring(_get(config.SEARCH_URL, params=params))
        entries = feed.findall("atom:entry", ATOM_NS)
        if not entries:
            break

        for entry in entries:
            ecli = entry.findtext("atom:id", default="", namespaces=ATOM_NS).strip()
            # Defensive: the feed occasionally carries entries whose id is not
            # an ECLI (service messages). Those are not rulings.
            if ecli.startswith("ECLI:"):
                collected.append(ecli)
            if limit is not None and len(collected) >= limit:
                return collected[:limit]

        offset += config.MAX_RESULTS_PER_PAGE

    return collected


def count_eclis(date_from: str, date_to: str, subject_uri: str | None = None) -> int:
    """Return how many rulings match a query, without downloading them all.

    The index feed reports the total match count in its ``<subtitle>`` element
    ("Aantal gevonden ECLI's: N"), so a single request with ``max=0`` is
    enough to size a stratum before deciding how much of it to sample.
    """
    params: list[tuple[str, str]] = [
        ("date", date_from),
        ("date", date_to),
        ("max", "1"),
    ]
    if config.REQUIRE_FULL_DOCUMENT:
        params.append(("return", "DOC"))
    if subject_uri is not None:
        params.append(("subject", subject_uri))

    feed = ET.fromstring(_get(config.SEARCH_URL, params=params))
    subtitle = feed.findtext("atom:subtitle", default="", namespaces=ATOM_NS)

    # Expected form: "Aantal gevonden ECLI's: 154833"
    digits = "".join(character for character in subtitle if character.isdigit())
    return int(digits) if digits else 0


def _cache_path(ecli: str) -> Path:
    """Return the on-disk location for a cached ruling.

    ECLIs contain colons, which are illegal in Windows filenames, so they are
    replaced by underscores. Files are bucketed by issuing court to keep any
    single directory from holding hundreds of thousands of entries, which
    makes both the filesystem and Explorer miserable.

    Example: ``ECLI:NL:RBDHA:2024:356`` -> ``raw/documents/RBDHA/ECLI_NL_RBDHA_2024_356.xml``
    """
    parts = ecli.split(":")
    court = parts[2] if len(parts) > 2 else "unknown"
    return config.RAW_DIR / "documents" / court / f"{ecli.replace(':', '_')}.xml"


def fetch_document(ecli: str, use_cache: bool = True) -> str:
    """Fetch one ruling's XML, reading from (and writing to) the disk cache.

    The cache is what makes the crawl restartable. Re-running the fetch step
    after an interruption costs a directory listing rather than another few
    hours of downloads.

    :param ecli: the ruling's ECLI identifier.
    :param use_cache: set to ``False`` to force a fresh download.
    :returns: the raw XML document as text.
    """
    path = _cache_path(ecli)

    if use_cache and path.exists():
        return path.read_text(encoding="utf-8")

    xml_text = _get(config.CONTENT_URL, params={"id": ecli})

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml_text, encoding="utf-8")
    return xml_text


def iter_cached_documents() -> Iterator[tuple[str, str]]:
    """Yield ``(ecli, xml_text)`` for every ruling already in the cache.

    Lets the parsing step run entirely offline, and lets a parsing bug be
    fixed and re-applied without touching the network again.
    """
    documents_root = config.RAW_DIR / "documents"
    if not documents_root.exists():
        return

    for path in sorted(documents_root.rglob("*.xml")):
        ecli = path.stem.replace("_", ":")
        yield ecli, path.read_text(encoding="utf-8")
