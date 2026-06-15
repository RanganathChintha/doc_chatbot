import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import requests
from requests.auth import HTTPBasicAuth

logger = logging.getLogger(__name__)

# A bare identity GUID, e.g. 8D45F413-0DC2-6FB3-A535-D794A26C509B
_GUID_RE = re.compile(
    r"[0-9A-Fa-f]{8}-(?:[0-9A-Fa-f]{4}-){3}[0-9A-Fa-f]{12}"
)
# An Azure DevOps person @mention: @<GUID> in raw markdown, or @GUID once
# the angle brackets have already been stripped.
_MENTION_RE = re.compile(r"@<?(" + _GUID_RE.pattern + r")>?")

# ══════════════════════════════════════════════════════════════════
#  SESSION
# ══════════════════════════════════════════════════════════════════

def _make_session(pat: str = "") -> requests.Session:
    """
    Build a requests.Session.
    PAT is sent as HTTP Basic auth (username='', password=PAT).
    This is the correct format for Azure DevOps, GitHub, Confluence, etc.
    Bearer token is NOT used — Azure DevOps rejects it.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/json,*/*;q=0.8",
    })
    if pat:
        session.auth = HTTPBasicAuth("", pat)
    return session

# ══════════════════════════════════════════════════════════════════
#  SHARED REST HELPER
# ══════════════════════════════════════════════════════════════════

def _rest_get(
    session: requests.Session,
    url:     str,
    params:  dict | None = None,
    timeout: int = 30,
    label:   str = "",
) -> requests.Response | None:
    """
    Perform a REST GET. Returns Response on success, None on any failure.
    Never raises — all errors are logged.
    """
    tag = f"[REST:{label}]" if label else "[REST]"
    try:
        resp = session.get(
            url,
            params=params or {},
            timeout=timeout,
            allow_redirects=True,
        )
        logger.debug("%s %s → HTTP %d", tag, url, resp.status_code)
        resp.raise_for_status()
        return resp
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        logger.error("%s HTTP %s → %s", tag, status, url)
    except requests.exceptions.ConnectionError as exc:
        logger.error("%s Connection error → %s: %s", tag, url, exc)
    except requests.exceptions.Timeout:
        logger.error("%s Timeout → %s", tag, url)
    except requests.exceptions.RequestException as exc:
        logger.error("%s Failed → %s: %s", tag, url, exc)
    return None

def _safe_json(resp: requests.Response, label: str) -> dict:
    """
    Parse JSON from a response.
    Guards against Azure DevOps returning an HTML login page instead of JSON
    when auth headers are missing or wrong.
    """
    ctype = resp.headers.get("Content-Type", "")
    if "text/html" in ctype:
        logger.error(
            "[%s] Got HTML instead of JSON (auth issue?). "
            "status=%d snippet=%s",
            label, resp.status_code, resp.text[:200],
        )
        return {}
    try:
        return resp.json()
    except Exception as exc:
        logger.error(
            "[%s] JSON decode failed: %s | body=%s", label, exc, resp.text[:200]
        )
        return {}

# ══════════════════════════════════════════════════════════════════
#  SHARED CONTENT HELPERS
# ══════════════════════════════════════════════════════════════════

def _resolve_urls(text: str, base_url: str) -> str:
    """Resolve relative URLs in markdown links to absolute URLs."""
    if not base_url or not text:
        return text
    base_dir = base_url.rstrip("/") + "/"

    def _replace(match: re.Match) -> str:
        href   = match.group(2)
        parsed = urlparse(href)
        if parsed.scheme or href.startswith(("mailto:", "#")):
            return match.group(0)
        return f"[{match.group(1)}]({urljoin(base_dir, href.lstrip('/'))})"

    return re.sub(r'\[([^\]]*)\]\(([^)]*)\)', _replace, text)

def _clean_markdown(text: str) -> str:
    """Strip images, HTML comments, raw HTML tags and extra blank lines."""
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r'@<([^>\s]+)>', r'@\1', text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ══════════════════════════════════════════════════════════════════
#  AZURE DEVOPS — IDENTITY (@MENTION) RESOLUTION
# ══════════════════════════════════════════════════════════════════

def _resolve_identities(
    session: requests.Session,
    org:     str,
    ids:     set[str],
) -> dict[str, str]:
    """
    Resolve a set of Azure DevOps identity GUIDs to display names.

    Azure DevOps stores wiki @mentions as @<GUID> where GUID is the user's
    identity id. The Identities API turns those ids into human names:

      GET https://vssps.dev.azure.com/{org}/_apis/identities
          ?identityIds={guid,guid,...}&api-version=7.1

    Returns { guid_lower: display_name }. Never raises — on any failure the
    affected GUIDs are simply left unresolved (the caller keeps the raw id).
    Requires the PAT to have Identity (Read) scope.
    """
    resolved: dict[str, str] = {}
    id_list = [i for i in ids if i]
    if not id_list:
        return resolved

    url = f"https://vssps.dev.azure.com/{org}/_apis/identities"

    # The API takes a comma-separated list; chunk to keep the URL sane.
    for start in range(0, len(id_list), 100):
        chunk = id_list[start:start + 100]
        resp = _rest_get(
            session, url,
            params={"identityIds": ",".join(chunk), "api-version": "7.1"},
            label="IDENTITY",
        )
        if resp is None:
            continue
        data = _safe_json(resp, "IDENTITY")
        for item in data.get("value", []) if data else []:
            if not isinstance(item, dict):
                continue  # API returns null for unresolvable ids
            gid  = (item.get("id") or "").lower()
            name = (
                item.get("providerDisplayName")
                or item.get("customDisplayName")
                or item.get("properties", {})
                    .get("Account", {}).get("$value", "")
                or item.get("properties", {})
                    .get("Mail", {}).get("$value", "")
            )
            if gid and name:
                resolved[gid] = name.strip()

    logger.info(
        "[AZURE-IDENTITY] Resolved %d/%d mentioned identit(y/ies).",
        len(resolved), len(id_list),
    )
    return resolved

def _collect_mention_ids(text: str) -> set[str]:
    """Return the lower-cased identity GUIDs @mentioned in *text*."""
    return {m.group(1).lower() for m in _MENTION_RE.finditer(text or "")}

def _apply_mentions(text: str, id_map: dict[str, str]) -> str:
    """Replace @<GUID> / @GUID mentions with @DisplayName where known."""
    if not text:
        return text

    def _replace(match: re.Match) -> str:
        name = id_map.get(match.group(1).lower())
        return f"@{name}" if name else match.group(0)

    return _MENTION_RE.sub(_replace, text)

def _extract_html_text(html: str, page_url: str) -> tuple[str, str]:
    """
    Parse raw HTML → (title, clean_text).
    Strips boilerplate and extracts main content area.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("title")
    title     = (
        title_tag.get_text(strip=True)
        if title_tag
        else urlparse(page_url).path.split("/")[-1] or page_url
    )

    for tag in soup(["script", "style", "nav", "header", "footer",
                     "aside", "form", "button", "iframe", "noscript"]):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", class_=re.compile(
            r"(wiki|main|page|post|entry|content|body)", re.I))
        or soup.find("div", id=re.compile(r"(main|content|page|body)", re.I))
        or soup.find("body")
    )

    raw   = main.get_text(separator="\n", strip=True) if main else ""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    text  = _resolve_urls("\n".join(lines), page_url)
    return title, text

# ══════════════════════════════════════════════════════════════════
#  PAGE TREE DISPLAY
# ══════════════════════════════════════════════════════════════════

def print_page_tree(pages: list[dict[str, Any]]) -> None:
    if not pages:
        print("  (no pages found)")
        return
    print("\n📚 Page Tree:")
    print("=" * 60)
    for page in pages:
        depth  = page.get("depth", 0)
        indent = "  " * depth
        is_container = any(
            p.get("path", "").startswith(page.get("path", "") + "/")
            for p in pages
            if p.get("path") != page.get("path")
        )
        icon  = "📁" if is_container else "📄"
        title = page.get("title") or page.get("path") or "?"
        print(f"{indent}{icon} {title}")
    print("=" * 60)
    print(f"Total: {len(pages)} page(s)\n")

# ══════════════════════════════════════════════════════════════════
#  AZURE DEVOPS — URL PARSER
# ══════════════════════════════════════════════════════════════════

def _parse_azure_wiki_url(wiki_url: str) -> dict[str, Any]:
    """
    Extract org, project, wiki_id, target_page_id, page_path
    from any Azure DevOps wiki browser URL.

    Supported formats
    ─────────────────
    https://dev.azure.com/Org/Project/_wiki/wikis/Wiki.wiki/
    https://dev.azure.com/Org/Project/_wiki/wikis/Wiki.wiki?pageId=19165
    https://dev.azure.com/Org/Project/_wiki/wikis/Wiki.wiki/19165/Page-Title
    https://dev.azure.com/Org/{guid}/_wiki/wikis/{guid}?pagePath=%2FSome%2FPage
    https://Org.visualstudio.com/Project/_wiki/wikis/Wiki.wiki?pageId=19165
    """
    parsed = urlparse(wiki_url.strip())
    host   = parsed.netloc.lower()
    parts  = [unquote(p) for p in parsed.path.split("/") if p]
    query  = parse_qs(parsed.query)

    def _qval(*names: str) -> str:
        for n in names:
            v = query.get(n)
            if v and v[0]:
                return v[0]
        return ""

    if host == "dev.azure.com":
        org, project = parts[0], parts[1]
    elif host.endswith(".visualstudio.com"):
        org, project = host.split(".")[0], parts[0]
    else:
        raise ValueError(f"Not an Azure DevOps URL: {wiki_url}")

    try:
        marker  = parts.index("wikis")
        wiki_id = parts[marker + 1] if len(parts) > marker + 1 else ""
    except ValueError:
        wiki_id = ""
    wiki_id = wiki_id or _qval("wikiIdentifier", "wiki_id", "wikiId")
    if not wiki_id:
        raise ValueError(f"Could not extract wiki_id from: {wiki_url}")

    pid_str = _qval("pageId", "page_id", "targetPageId")
    if not pid_str:
        for part in parts[marker + 2:] if marker >= 0 else []:
            if part.isdigit():
                pid_str = part
                break

    page_path = _qval("pagePath")

    return {
        "organization":   org,
        "project":        project,
        "wiki_id":        wiki_id,
        "target_page_id": int(pid_str) if pid_str and pid_str.isdigit() else None,
        "page_path":      page_path,
    }

def _path_to_title(path: str) -> str:
    return (
        path.split("/")[-1].replace("-", " ").replace("_", " ").strip()
        if path else ""
    )

# ══════════════════════════════════════════════════════════════════
#  AZURE DEVOPS WIKI — COMPLETE PAGE DISCOVERY VIA PAGESBATCH
# ══════════════════════════════════════════════════════════════════

def _fetch_all_wiki_pages(
    session: requests.Session,
    base:    str,
    org:     str,
    project: str,
    wiki_id: str,
    ver:     str,
) -> list[dict[str, Any]]:
    """
    Fetch EVERY page in the wiki using the pagesBatch endpoint.

    Why not recursionLevel=full?
    ────────────────────────────
    The Azure DevOps wiki pages API with recursionLevel only returns
    1 level of subPages per call regardless of the value passed.
    The only reliable way to get ALL pages is the pagesBatch endpoint:

      POST /_apis/wiki/wikis/{wikiId}/pagesBatch

    This returns every page in one shot, paginated via
    x-ms-continuationtoken if there are more than 100 pages.
    """
    batch_url           = base.replace("/pages", "/pagesBatch")
    all_pages:          list[dict[str, Any]] = []
    continuation_token: str | None           = None

    while True:
        headers = {"Content-Type": "application/json"}
        if continuation_token:
            headers["x-ms-continuationtoken"] = continuation_token

        try:
            resp = session.post(
                batch_url,
                params={"api-version": ver},
                headers=headers,
                json={
                    "top":               100,
                    "continuationToken": continuation_token,
                },
                timeout=60,
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.error("[AZURE-BATCH] pagesBatch failed: %s", exc)
            break

        data  = _safe_json(resp, "AZURE-BATCH")
        items = data.get("value", []) if data else []

        if not items:
            break

        for item in items:
            page_path = item.get("path", "") or ""
            page_id   = item.get("id")
            depth     = max(page_path.count("/") - 1, 0)

            all_pages.append({
                "id":          page_id,
                "path":        page_path,
                "title":       _path_to_title(page_path),
                "content":     "",
                "source":      f"Azure Wiki - {project}/{wiki_id}{page_path}",
                "source_type": "wiki",
                "remote_url":  item.get("remoteUrl", "") or (
                    f"https://dev.azure.com/{org}/{project}"
                    f"/_wiki/wikis/{wiki_id}/{page_id}"
                    if page_id is not None else ""
                ),
                "depth":       depth,
                "char_count":  0,
            })

        continuation_token = resp.headers.get("x-ms-continuationtoken")
        if not continuation_token:
            break

        logger.info(
            "[AZURE-BATCH] Fetched %d page(s) so far, continuing...",
            len(all_pages),
        )

    logger.info(
        "[AZURE-BATCH] Discovered %d total page(s) in wiki '%s'.",
        len(all_pages), wiki_id,
    )
    return all_pages

def _filter_pages_by_path(
    pages:     list[dict[str, Any]],
    root_path: str,
) -> list[dict[str, Any]]:
    """
    Filter pages to only those under root_path.
    Recalculates depth relative to root_path.
    """
    root     = root_path.rstrip("/")
    filtered = [
        p for p in pages
        if p["path"] == root or p["path"].startswith(root + "/")
    ]
    root_depth = root.count("/")
    for p in filtered:
        p["depth"] = p["path"].count("/") - root_depth
    return filtered

def _discover_azure_pages(
    session:        requests.Session,
    base:           str,
    org:            str,
    project:        str,
    wiki_id:        str,
    ver:            str,
    target_page_id: int | None,
    page_path:      str,
) -> list[dict[str, Any]]:
    """
    Discover all pages to fetch.

    Always fetches the FULL wiki page list via pagesBatch first,
    then filters down if a specific page/path was requested.

    Priority
    ────────
    1. pagePath  → filter full list to that subtree
    2. page ID   → resolve path from full list → filter to subtree
    3. neither   → return all pages
    """
    all_pages = _fetch_all_wiki_pages(
        session, base, org, project, wiki_id, ver,
    )

    if not all_pages:
        return []

    # ── Case 1: pagePath ───────────────────────────────────────────
    if page_path:
        filtered = _filter_pages_by_path(all_pages, page_path)
        logger.info(
            "[AZURE] pagePath='%s' → %d page(s) in subtree.",
            page_path, len(filtered),
        )
        return filtered if filtered else all_pages

    # ── Case 2: numeric page ID ────────────────────────────────────
    if target_page_id is not None:
        match = next(
            (p for p in all_pages if p.get("id") == target_page_id), None
        )
        if match:
            filtered = _filter_pages_by_path(all_pages, match["path"])
            logger.info(
                "[AZURE] pageId=%d → path='%s' → %d page(s) in subtree.",
                target_page_id, match["path"], len(filtered),
            )
            return filtered
        logger.warning(
            "[AZURE] pageId=%d not found — returning all pages.", target_page_id
        )

    # ── Case 3: entire wiki ────────────────────────────────────────
    logger.info("[AZURE] Returning all %d page(s).", len(all_pages))
    return all_pages

# ══════════════════════════════════════════════════════════════════
#  AZURE DEVOPS WIKI — CONTENT HYDRATION
# ══════════════════════════════════════════════════════════════════

def _hydrate_azure_pages(
    pages:   list[dict[str, Any]],
    session: requests.Session,
    base:    str,
    org:     str,
    project: str,
    wiki_id: str,
    ver:     str,
) -> list[dict[str, Any]]:
    """
    Fetch full Markdown content for every page concurrently.
    Tries path-based fetch first, falls back to ID-based fetch.
    """
    wiki_root = f"https://dev.azure.com/{org}/{project}/_wiki/wikis/{wiki_id}/"

    def _hydrate(page: dict[str, Any]) -> dict[str, Any]:
        content = ""

        # Primary: fetch by path
        resp = _rest_get(
            session, base,
            params={
                "path":           page["path"],
                "includeContent": "true",
                "api-version":    ver,
            },
            timeout=30,
            label=f"HYDRATE:{page.get('title', '')}",
        )
        if resp is not None:
            data    = _safe_json(resp, "HYDRATE")
            content = (data.get("content", "") or "") if data else ""

        # Fallback: fetch by ID
        if not content and page.get("id") is not None:
            resp = _rest_get(
                session,
                f"{base}/{page['id']}",
                params={"includeContent": "true", "api-version": ver},
                timeout=30,
                label=f"HYDRATE-ID:{page.get('title', '')}",
            )
            if resp is not None:
                data    = _safe_json(resp, "HYDRATE-ID")
                content = (data.get("content", "") or "") if data else ""

        page["_raw"] = content
        return page

    with ThreadPoolExecutor(max_workers=8) as pool:
        pages = list(pool.map(_hydrate, pages))

    # Resolve every @mention GUID across all pages in a single batched call,
    # then substitute display names before cleaning/url-resolution.
    mention_ids: set[str] = set()
    for page in pages:
        mention_ids |= _collect_mention_ids(page.get("_raw", ""))
    id_map = _resolve_identities(session, org, mention_ids) if mention_ids else {}

    for page in pages:
        raw                = page.pop("_raw", "")
        raw                = _apply_mentions(raw, id_map)
        clean              = _clean_markdown(raw)
        clean              = _resolve_urls(clean, wiki_root)
        page["content"]    = clean
        page["char_count"] = len(clean)

    with_content = sum(1 for p in pages if p.get("content"))
    logger.info(
        "[AZURE] Hydrated %d page(s): %d with content, %d empty.",
        len(pages), with_content, len(pages) - with_content,
    )
    return pages

# ══════════════════════════════════════════════════════════════════
#  AZURE DEVOPS WIKI — FULL PIPELINE
# ══════════════════════════════════════════════════════════════════

def _fetch_azure_wiki(pat: str, wiki_url: str) -> list[dict[str, Any]]:
    """
    Full Azure DevOps wiki fetch pipeline via REST API.

    1. Parse URL → extract org, project, wiki_id, page_id / pagePath
    2. Build authenticated session (Basic auth with PAT)
    3. Verify PAT works before doing anything
    4. Discover all pages via pagesBatch (handles pagination)
    5. Filter to subtree if page_id or pagePath was specified
    6. Hydrate every page with its Markdown content concurrently
    """
    params         = _parse_azure_wiki_url(wiki_url)
    org            = params["organization"]
    project        = params["project"]
    wiki_id        = params["wiki_id"]
    target_page_id = params["target_page_id"]
    page_path      = params["page_path"]
    ver            = "7.1"
    base           = (
        f"https://dev.azure.com/{org}/{project}"
        f"/_apis/wiki/wikis/{wiki_id}/pages"
    )

    logger.info(
        "[AZURE] org=%s project=%s wiki_id=%s page_id=%s page_path=%s",
        org, project, wiki_id, target_page_id, page_path or "(none)",
    )

    if not pat:
        raise PermissionError(
            "Azure DevOps wiki requires a Personal Access Token (PAT). "
            "Pass pat='your-token' to fetch_pages_from_url()."
        )

    session = _make_session(pat=pat)

    # Verify PAT before crawling
    probe = session.get(
        base,
        params={"path": "/", "api-version": ver},
        timeout=15,
    )
    if probe.status_code == 401:
        raise PermissionError(
            f"Azure DevOps authentication failed (HTTP 401). "
            f"The PAT is invalid or expired. wiki_id={wiki_id}"
        )
    if probe.status_code == 403:
        raise PermissionError(
            f"Azure DevOps access denied (HTTP 403). "
            f"The PAT lacks Wiki (Read) permission. wiki_id={wiki_id}"
        )

    pages = _discover_azure_pages(
        session, base, org, project, wiki_id, ver,
        target_page_id=target_page_id,
        page_path=page_path,
    )

    if not pages:
        raise ValueError(
            f"No pages found in wiki '{wiki_id}'. "
            "Check the URL and PAT permissions."
        )

    return _hydrate_azure_pages(
        pages, session, base, org, project, wiki_id, ver,
    )

# ══════════════════════════════════════════════════════════════════
#  AZURE DEVOPS — GENERIC REST API FETCHER (non-wiki URLs)
# ══════════════════════════════════════════════════════════════════

def _normalise_azure_response(
    data:       dict | list,
    source_url: str,
) -> list[dict[str, Any]]:
    """
    Convert any Azure DevOps REST API JSON response into a
    normalised list of page dicts.

    Handles:
      - { "value": [ ... ] }  — paginated list responses
      - { ... }               — single object response
      - [ ... ]               — raw list response
    """
    items: list[dict] = []

    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        if "value" in data:
            items = data["value"]
        else:
            items = [data]

    pages: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        title = (
            item.get("name")
            or item.get("title")
            or item.get("id")
            or item.get("path")
            or f"item-{i}"
        )
        content    = json.dumps(item, indent=2, default=str)
        remote_url = (
            item.get("remoteUrl")
            or item.get("url")
            or item.get("webUrl")
            or item.get("_links", {}).get("web", {}).get("href", "")
            or source_url
        )
        pages.append({
            "id":          item.get("id") or f"{source_url}#{i}",
            "path":        item.get("path") or f"/{title}",
            "title":       str(title),
            "content":     content,
            "source":      source_url,
            "source_type": "azure",
            "remote_url":  remote_url,
            "depth":       0,
            "char_count":  len(content),
        })

    logger.info(
        "[AZURE] Normalised %d item(s) from %s.", len(pages), source_url
    )
    return pages

def _fetch_azure_rest_url(pat: str, url: str) -> list[dict[str, Any]]:
    """
    Fetch any Azure DevOps URL via REST API.

    Routing
    ───────
    /_wiki/wikis/ URL  →  _fetch_azure_wiki()  (full tree + markdown)
    any other URL      →  direct REST GET, JSON normalised to page dicts
    """
    if not pat:
        raise PermissionError(
            "Azure DevOps requires a Personal Access Token (PAT). "
            "Pass pat='your-token' to fetch_pages_from_url()."
        )

    parsed  = urlparse(url)
    is_wiki = "/_wiki/wikis/" in parsed.path

    if is_wiki:
        return _fetch_azure_wiki(pat=pat, wiki_url=url)

    # ── Generic Azure REST API URL ─────────────────────────────────
    session = _make_session(pat=pat)

    probe = session.get(url, timeout=15)
    if probe.status_code == 401:
        raise PermissionError(
            f"Azure DevOps authentication failed (HTTP 401). "
            f"The PAT is invalid or expired. url={url}"
        )
    if probe.status_code == 403:
        raise PermissionError(
            f"Azure DevOps access denied (HTTP 403). "
            f"The PAT lacks permission for this resource. url={url}"
        )

    if "/_apis/" not in parsed.path:
        logger.warning(
            "[AZURE] URL does not contain /_apis/ — attempting direct fetch: %s", url
        )

    resp = _rest_get(
        session, url,
        params={"api-version": "7.1"},
        label="AZURE-GENERIC",
    )
    if resp is None:
        raise ValueError(f"Failed to fetch Azure DevOps URL: {url}")

    data = _safe_json(resp, "AZURE-GENERIC")
    if not data:
        raise ValueError(f"Empty or invalid JSON response from: {url}")

    return _normalise_azure_response(data, url)

# ══════════════════════════════════════════════════════════════════
#  GENERIC HTML CRAWLER — for any non-Azure URL
# ══════════════════════════════════════════════════════════════════

def _resolve_generic_session(url: str, pat: str = "") -> requests.Session:
    """
    Try anonymous first. If 401/403 and PAT provided → use PAT session.
    Raises PermissionError if auth fails or is missing.
    """
    anon = _make_session()
    try:
        resp = anon.get(url, timeout=15, allow_redirects=True)
        if resp.status_code not in (401, 403):
            logger.info("[HTML] %s → HTTP %d (anonymous OK).", url, resp.status_code)
            return anon
        logger.info("[HTML] %s → HTTP %d (auth required).", url, resp.status_code)
    except requests.exceptions.RequestException as exc:
        if not pat:
            raise ConnectionError(
                f"Could not reach {url}. Check the URL and network."
            ) from exc

    if not pat:
        raise PermissionError(
            f"{url} requires authentication. Provide a PAT."
        )

    pat_session = _make_session(pat=pat)
    resp        = pat_session.get(url, timeout=15, allow_redirects=True)
    if resp.status_code == 401:
        raise PermissionError(
            f"Authentication failed for {url} (HTTP 401). PAT is invalid."
        )
    if resp.status_code == 403:
        raise PermissionError(
            f"Access denied for {url} (HTTP 403). PAT lacks permission."
        )

    logger.info("[HTML] PAT auth succeeded for %s (HTTP %d).", url, resp.status_code)
    return pat_session

def _fetch_html_page(
    session: requests.Session,
    url:     str,
    depth:   int = 0,
) -> dict[str, Any]:
    resp = _rest_get(session, url, label="PAGE")
    if resp is None or "text/html" not in resp.headers.get("Content-Type", ""):
        parsed = urlparse(url)
        return {
            "id":          url,
            "path":        parsed.path,
            "title":       parsed.path.split("/")[-1] or url,
            "content":     "",
            "source":      url,
            "source_type": "web",
            "remote_url":  url,
            "depth":       depth,
            "char_count":  0,
        }

    title, text = _extract_html_text(resp.text, url)
    parsed      = urlparse(url)
    return {
        "id":          url,
        "path":        parsed.path,
        "title":       title,
        "content":     text,
        "source":      url,
        "source_type": "web",
        "remote_url":  url,
        "depth":       depth,
        "char_count":  len(text),
    }

def _crawl_html(
    session:   requests.Session,
    start_url: str,
    max_pages: int = 0,
    max_depth: int = 0,
) -> list[tuple[str, int]]:
    """BFS crawl — same domain, same path prefix, authenticated session."""
    parsed_start = urlparse(start_url)
    base_domain  = parsed_start.netloc.lower()
    base_path    = parsed_start.path.rstrip("/")

    visited:    set[str]              = set()
    queue:      list[tuple[str, int]] = [(start_url, 0)]
    discovered: list[tuple[str, int]] = []

    while queue:
        if max_pages > 0 and len(discovered) >= max_pages:
            break

        url, depth = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        discovered.append((url, depth))

        if max_depth > 0 and depth >= max_depth:
            continue

        resp = _rest_get(session, url, label="CRAWL")
        if resp is None or "text/html" not in resp.headers.get("Content-Type", ""):
            continue

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")

        for anchor in soup.find_all("a", href=True):
            href = anchor["href"].strip()
            if not href or href.startswith(("#", "mailto:", "javascript:", "tel:")):
                continue
            absolute = urljoin(url, href)
            parsed   = urlparse(absolute)
            if parsed.scheme not in ("http", "https"):
                continue
            if parsed.netloc.lower() != base_domain:
                continue
            if not parsed.path.startswith(base_path):
                continue
            if parsed.path in (base_path, base_path + "/"):
                continue
            clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if clean not in visited:
                queue.append((clean, depth + 1))

    logger.info(
        "[HTML] Crawl done: %d page(s) from %s.", len(discovered), start_url,
    )
    return discovered

def _fetch_generic_site(
    url:       str,
    pat:       str = "",
    max_pages: int = 0,
    max_depth: int = 0,
) -> list[dict[str, Any]]:
    """HTML BFS crawler for any non-Azure website."""
    session         = _resolve_generic_session(url, pat=pat)
    url_depth_pairs = _crawl_html(
        session, url, max_pages=max_pages, max_depth=max_depth,
    )

    with ThreadPoolExecutor(max_workers=8) as pool:
        pages = list(pool.map(
            lambda pair: _fetch_html_page(session, pair[0], depth=pair[1]),
            url_depth_pairs,
        ))

    with_content = sum(1 for p in pages if p.get("content"))
    logger.info(
        "[HTML] Done: %d page(s), %d with content, %d empty.",
        len(pages), with_content, len(pages) - with_content,
    )
    return pages

# ══════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def fetch_pages_from_url(
    url:       str,
    pat:       str = "",
    max_pages: int = 0,
    max_depth: int = 0,
) -> list[dict[str, Any]]:
    """
    Fetch *url* and every child page. Works for any website.

    Routing
    ───────
    dev.azure.com or *.visualstudio.com
      ├─ /_wiki/wikis/ URL  →  Azure Wiki REST API
      │                         - pagesBatch for full page discovery
      │                         - pagination via x-ms-continuationtoken
      │                         - concurrent Markdown hydration
      └─ any other URL      →  Azure Generic REST API
                                - direct JSON fetch + normalise

    Any other domain        →  HTML BFS crawler
                                - anonymous or PAT Basic auth
                                - same domain + path prefix only

    Parameters
    ----------
    url       : Any HTTP/HTTPS URL.
    pat       : PAT / password. Required for Azure DevOps.
                Optional for other private sites.
    max_pages : Max pages to crawl. 0 = unlimited. (HTML crawler only)
    max_depth : Max link depth.    0 = unlimited. (HTML crawler only)
    """
    if not url:
        raise ValueError("url must not be empty.")

    parsed = urlparse(url)
    host   = parsed.netloc.lower()

    is_azure = (
        host == "dev.azure.com"
        or host.endswith(".visualstudio.com")
    )

    if is_azure:
        logger.info("[ROUTER] Azure DevOps URL → REST API backend.")
        pages = _fetch_azure_rest_url(pat=pat, url=url)
    else:
        logger.info("[ROUTER] Generic URL → HTML crawler backend.")
        pages = _fetch_generic_site(
            url=url,
            pat=pat,
            max_pages=max_pages,
            max_depth=max_depth,
        )

    print_page_tree(pages)
    return pages
