import logging
import os
from collections import deque
from hashlib import md5
from pathlib import PurePosixPath
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from app.langsmith_tracing import langsmith_traceable as traceable

logger = logging.getLogger(__name__)

_TEXT_SELECTOR = ["article", "main", "section", "div", "p", "h1", "h2", "h3", "h4"]
_NON_HTML_EXTENSIONS = {
    ".7z",
    ".avi",
    ".bmp",
    ".css",
    ".csv",
    ".doc",
    ".docx",
    ".gif",
    ".gz",
    ".ico",
    ".jpeg",
    ".jpg",
    ".js",
    ".json",
    ".mov",
    ".mp3",
    ".mp4",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".rar",
    ".svg",
    ".tar",
    ".tif",
    ".tiff",
    ".webp",
    ".xls",
    ".xlsx",
    ".xml",
    ".zip",
}


def _normalize_url(base_url: str, link: str) -> str | None:
    try:
        joined = urljoin(base_url, link)
        parsed = urlparse(joined)
        if parsed.scheme not in {"http", "https"}:
            return None

        scheme = parsed.scheme.lower()
        host = (parsed.hostname or "").lower()
        if not host:
            return None
        if parsed.port and not (
            (scheme == "http" and parsed.port == 80)
            or (scheme == "https" and parsed.port == 443)
        ):
            netloc = f"{host}:{parsed.port}"
        else:
            netloc = host

        path = parsed.path or "/"
        if path != "/":
            path = path.rstrip("/")
        query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))
        return urlunparse((scheme, netloc, path, "", query, ""))
    except Exception:
        return None


def _host_from_domain(value: str) -> str | None:
    parsed = urlparse(value if "://" in value else f"https://{value}")
    return (parsed.hostname or "").lower() or None


def _is_probable_html_url(url: str) -> bool:
    path = urlparse(url).path
    suffix = PurePosixPath(path).suffix.lower()
    return suffix not in _NON_HTML_EXTENSIONS


def _allowed_host(raw_host: str, allowed_hosts: set[str]) -> bool:
    if not raw_host:
        return False
    host = raw_host.lower()
    return any(host == allowed or host.endswith('.' + allowed) for allowed in allowed_hosts)


def _extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "header", "footer", "nav"]):
        tag.decompose()

    content_blocks = []
    for selector in ["article", "main", "section"]:
        section = soup.find(selector)
        if section:
            content_blocks.append(section)
    if not content_blocks:
        content_blocks = soup.find_all(_TEXT_SELECTOR)

    texts = []
    for block in content_blocks:
        text = block.get_text(" ", strip=True)
        if text and len(text) > 40:
            texts.append(text)

    if not texts:
        return soup.get_text(" ", strip=True)
    return "\n\n".join(texts)


def _extract_links(html: str, base_url: str, allowed_hosts: set[str]) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    seen_links: set[str] = set()
    for element in soup.find_all("a", href=True):
        normalized = _normalize_url(base_url, element["href"])
        if not normalized:
            continue
        if not _is_probable_html_url(normalized):
            continue
        parsed = urlparse(normalized)
        if _allowed_host(parsed.netloc, allowed_hosts) and normalized not in seen_links:
            seen_links.add(normalized)
            links.append(normalized)
    return links


def _page_signature(text: str, title: str) -> str:
    value = f"{title}\n{text}"
    return md5(value.encode("utf-8"), usedforsecurity=False).hexdigest()


class _PlaywrightRenderer:
    """Keeps a single headless browser open for the duration of one crawl.

    Launching a fresh browser per page (~7s of startup each) makes multi-page
    crawls pathologically slow, so the browser/context are created once and
    every page is rendered through a reused context. Construction raises if
    Playwright isn't usable, letting the caller fall back to plain HTTP upfront.
    """

    def __init__(
        self,
        headers: dict[str, str] | None,
        auth_cookies: dict[str, str] | None,
        cookie_url: str | None,
        timeout: int,
    ) -> None:
        if os.name == "nt":
            try:
                import asyncio

                policy = getattr(asyncio, "WindowsProactorEventLoopPolicy", None)
                if policy is not None:
                    asyncio.set_event_loop_policy(policy())
            except Exception:
                logger.debug("Unable to set Windows Proactor event loop policy", exc_info=True)

        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is not installed; install playwright to enable JS rendering"
            ) from exc

        self._timeout = timeout
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=True)
        self._context = self._browser.new_context()
        if headers:
            self._context.set_extra_http_headers(headers)
        if auth_cookies and cookie_url:
            self._context.add_cookies(
                [{"name": n, "value": v, "url": cookie_url} for n, v in auth_cookies.items()]
            )

    def render(self, url: str) -> str:
        page = self._context.new_page()
        try:
            # "domcontentloaded" returns as soon as the DOM is parsed (after on-load
            # JS), which is fast and reliable. "networkidle" waits for 500ms of zero
            # network traffic — many sites (analytics, fonts, polling) never reach it,
            # so waiting on it would burn the full timeout on every page.
            page.goto(url, wait_until="domcontentloaded", timeout=self._timeout * 1000)
            try:
                # Best-effort settle for client-rendered content; never block the
                # whole render budget on it.
                page.wait_for_load_state("networkidle", timeout=3000)
            except Exception:
                pass
            return page.content()
        finally:
            page.close()

    def close(self) -> None:
        try:
            self._browser.close()
        finally:
            self._pw.stop()


@traceable(run_type="tool", name="load_url_documents")
def load_url_documents(
    start_urls: list[str],
    allow_domains: list[str] | None = None,
    max_pages: int | None = 20,
    max_depth: int | None = 2,
    auth_cookies: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    render_javascript: bool = False,
    render_timeout: int = 30,
) -> list[Document]:
    """Crawl one or more authenticated internal URLs and return text Documents."""
    if not start_urls:
        return []

    allowed_hosts = {host for host in (_host_from_domain(domain) for domain in (allow_domains or [])) if host}
    normalized_starts: list[str] = []
    for url in start_urls:
        normalized = _normalize_url(url, url)
        if not normalized:
            continue
        normalized_starts.append(normalized)
        parsed = urlparse(normalized)
        if parsed.netloc:
            allowed_hosts.add(parsed.netloc.lower())
    if not allowed_hosts:
        raise ValueError("No valid URL hosts were found.")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "doc-chatbot-crawler/1.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })
    if headers:
        session.headers.update(headers)
    if auth_cookies:
        session.cookies.update(auth_cookies)

    docs: list[Document] = []
    seen: set[str] = set()
    queued: set[str] = set(normalized_starts)
    content_signatures: set[str] = set()
    queue = deque((url, 0) for url in normalized_starts)
    # One browser for the whole crawl (see _PlaywrightRenderer). If Playwright
    # isn't usable, fall back to plain HTTP for the entire crawl upfront so we
    # don't repeatedly try (and fail) to launch a browser per page.
    renderer = None
    if render_javascript:
        try:
            renderer = _PlaywrightRenderer(
                headers,
                auth_cookies,
                normalized_starts[0] if normalized_starts else None,
                render_timeout,
            )
        except Exception as exc:
            logger.warning(
                "Playwright unavailable (%s); using direct HTTP requests for this crawl.",
                exc,
            )
            renderer = None

    try:
        while queue and (max_pages is None or len(seen) < max_pages):
            url, depth = queue.popleft()
            if url in seen or (max_depth is not None and depth > max_depth):
                continue
            seen.add(url)

            html = None
            if renderer is not None and _is_probable_html_url(url):
                try:
                    html = renderer.render(url)
                except Exception as exc:
                    # Per-page render failure (e.g. timeout): fall back to HTTP for
                    # just this page; the browser stays open for the rest of the crawl.
                    logger.warning("Playwright render failed for %s: %s", url, exc)
                    html = None

            if html is None:
                try:
                    resp = session.get(url, timeout=render_timeout)
                    if resp.status_code != 200:
                        logger.warning("URL loader skipped %s — status %s", url, resp.status_code)
                        continue
                    if "text/html" not in resp.headers.get("Content-Type", ""):
                        logger.debug("URL loader skipped non-HTML %s", url)
                        continue
                    html = resp.text
                except Exception as exc:
                    logger.warning("URL loader request failed for %s: %s", url, exc)
                    continue

            text = _extract_text(html)
            if not text or len(text) < 50:
                logger.debug("URL loader found no text for %s", url)
            title = ""
            try:
                soup = BeautifulSoup(html, "html.parser")
                title = soup.title.string.strip() if soup.title and soup.title.string else ""
            except Exception:
                title = ""

            signature = _page_signature(text, title)
            if signature in content_signatures:
                logger.debug("URL loader skipped duplicate content at %s", url)
            else:
                content_signatures.add(signature)
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": url,
                        "source_type": "web",
                        "title": title,
                        "signature": signature,
                    },
                ))

            if max_depth is None or depth < max_depth:
                for link in _extract_links(html, url, allowed_hosts):
                    if link not in seen and link not in queued and (
                        max_pages is None or len(seen) + len(queue) < max_pages
                    ):
                        queued.add(link)
                        queue.append((link, depth + 1))
    finally:
        if renderer is not None:
            renderer.close()

    logger.info(
        "URL loader crawled %d page(s) from %d start URL(s)",
        len(docs),
        len(start_urls),
    )
    return docs
