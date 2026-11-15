"""Breadth-first crawler: visits the pages of one site and collects their image URLs."""
import logging
import re
import time
from collections import deque
from collections.abc import Iterator
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

USER_AGENT = "face-search-crawler/1.0"
SKIPPED_EXTENSIONS = (".pdf", ".zip", ".mp4")
LAZY_SRC_ATTRS = ("data-src", "data-lazy-src", "data-original")
SRCSET_DESCRIPTOR = re.compile(r"(\d+(?:\.\d+)?)[wx]")


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def normalize_url(url: str, base_url: str) -> str:
    """Relative -> absolute, and drop the #fragment (same resource either way)."""
    absolute, _fragment = urldefrag(urljoin(base_url, url.strip()))
    return absolute


def same_site(url: str, start_url: str) -> bool:
    def host(u: str) -> str:
        return urlparse(u).netloc.lower().removeprefix("www.")
    return host(url) == host(start_url)


def largest_from_srcset(srcset: str) -> str | None:
    """'a.jpg 400w, b.jpg 800w' -> 'b.jpg' (also handles '1x, 2x' descriptors).

    URLs may contain commas themselves (CNN: '...?q=h_144,w_256,c_fill 256w'),
    so we can't split on ','. Instead we read whitespace-separated tokens:
    a token that looks like '800w' or '2x' is the descriptor of the URL before it.
    """
    candidates = []  # (url, size)
    url = None
    for token in srcset.split():
        descriptor = SRCSET_DESCRIPTOR.fullmatch(token.rstrip(","))
        if descriptor and url:
            candidates.append((url, float(descriptor.group(1))))
            url = None
        else:
            if url:  # previous URL had no descriptor: defaults to "1x"
                candidates.append((url, 1.0))
            url = token.strip(",")
    if url:
        candidates.append((url, 1.0))
    return max(candidates, key=lambda c: c[1])[0] if candidates else None


def extract_image_urls(soup: BeautifulSoup, page_url: str) -> set[str]:
    raw = []
    for img in soup.find_all("img"):
        raw.append(img.get("src"))
        raw.extend(img.get(attr) for attr in LAZY_SRC_ATTRS)
        if img.get("srcset"):
            raw.append(largest_from_srcset(img["srcset"]))
    for source in soup.select("picture source[srcset]"):
        raw.append(largest_from_srcset(source["srcset"]))
    for meta in soup.select('meta[property="og:image"]'):
        raw.append(meta.get("content"))

    urls = {normalize_url(u, page_url) for u in raw if u}
    # Drops inline data: URIs and anything that isn't a downloadable http(s) URL
    return {u for u in urls if urlparse(u).scheme in ("http", "https")}


def extract_links(soup: BeautifulSoup, page_url: str, start_url: str) -> set[str]:
    links = {normalize_url(a["href"], page_url) for a in soup.find_all("a", href=True)}
    return {
        u for u in links
        if urlparse(u).scheme in ("http", "https")
        and same_site(u, start_url)
        and not urlparse(u).path.lower().endswith(SKIPPED_EXTENSIONS)
    }


def load_robots(start_url: str) -> RobotFileParser:
    robots = RobotFileParser(urljoin(start_url, "/robots.txt"))
    try:
        robots.read()
    except OSError:
        log.warning("Could not read robots.txt, assuming everything is allowed")
        robots.parse([])
    return robots


def crawl(start_url: str, max_pages: int, delay_s: float,
          session: requests.Session) -> Iterator[tuple[str, set[str]]]:
    """Yields (page_url, image_urls) for each HTML page reached from start_url."""
    robots = load_robots(start_url)
    queue = deque([normalize_url(start_url, start_url)])
    seen = set(queue)
    visited = 0

    while queue and visited < max_pages:
        url = queue.popleft()
        if not robots.can_fetch(USER_AGENT, url):
            log.info("Disallowed by robots.txt: %s", url)
            continue
        try:
            # stream=True: only the headers are fetched here, the body is read on demand
            with session.get(url, timeout=15, stream=True) as response:
                response.raise_for_status()
                if "text/html" not in response.headers.get("Content-Type", ""):
                    continue  # leaving the "with" closes the connection: body never downloaded
                html = response.text
                # Use the final URL: redirects change the base for relative links
                page_url = normalize_url(response.url, response.url)
        except requests.RequestException as exc:
            log.warning("Skipping %s: %s", url, exc)
            continue
        finally:
            time.sleep(delay_s)  # politeness: don't hammer the site

        visited += 1
        soup = BeautifulSoup(html, "html.parser")

        image_urls = extract_image_urls(soup, page_url)
        log.info("[%d/%d] %s -> %d images", visited, max_pages, page_url, len(image_urls))
        yield page_url, image_urls

        for link in extract_links(soup, page_url, start_url) - seen:
            seen.add(link)
            queue.append(link)
