import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import requests
from requests.auth import HTTPBasicAuth

logger = logging.getLogger(__name__)

AZURE_DEVOPS_ORG = os.getenv("AZURE_DEVOPS_ORG", "")
AZURE_DEVOPS_PROJECT = os.getenv("AZURE_DEVOPS_PROJECT", "")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "7.1")

# ──────────────────────────────────────────────
#  Azure DevOps REST API helpers
# ──────────────────────────────────────────────

def parse_wiki_url(wiki_url: str) -> dict[str, str | int]:
    """
    Extract Azure DevOps wiki fetch parameters from a browser URL.

    Handles all real-world Azure DevOps wiki URL formats:

    Format 1 — wiki root only (NO page ID) ← your case
      https://dev.azure.com/Org/Project/_wiki/wikis/Wiki.wiki/
      → target_page_id = None  (signals: fetch entire wiki)

    Format 2 — pageId in query string (standard browser URL)
      https://dev.azure.com/Org/Project/_wiki/wikis/Wiki.wiki?pageId=19165

    Format 3 — numeric page ID in path segments
      https://dev.azure.com/Org/Project/_wiki/wikis/Wiki.wiki/19165/Digital-Commerce-OPS

    Format 4 — visualstudio.com legacy
      https://Org.visualstudio.com/Project/_wiki/wikis/Wiki.wiki?pageId=19165
    """
    parsed = urlparse(wiki_url.strip())
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Enter a complete Azure DevOps wiki URL.")

    host  = parsed.netloc.lower()
    parts = [unquote(p) for p in parsed.path.split("/") if p]
    query = parse_qs(parsed.query)

    # ── Extract org + project ──────────────────────────────────────
    organization = ""
    project      = ""

    if host == "dev.azure.com":
        if len(parts) < 2:
            raise ValueError("Azure DevOps URL must include organization and project.")
        organization = parts[0]
        project      = parts[1]

    elif host.endswith(".visualstudio.com"):
        organization = parsed.netloc.split(".")[0]
        if not parts:
            raise ValueError("Azure DevOps URL must include a project.")
        project = parts[0]

    else:
        raise ValueError("URL must be from dev.azure.com or visualstudio.com.")

    # ── Extract wiki_id ────────────────────────────────────────────
    wiki_id = _first_query_value(query, "wikiIdentifier", "wiki_id", "wikiId")

    try:
        marker = parts.index("wikis")
    except ValueError:
        marker = -1

    if marker >= 0 and len(parts) > marker + 1:
        wiki_id = wiki_id or parts[marker + 1]

    if not wiki_id:
        raise ValueError(
            "Could not find wiki id in the URL. "
            "Expected format: .../_wiki/wikis/{wikiId} or .../_wiki/wikis/{wikiId}?pageId={id}"
        )

    # ── Extract target_page_id (OPTIONAL) ─────────────────────────
    # Priority:
    #   1. ?pageId=NNN       in query string
    #   2. ?page_id=NNN      in query string
    #   3. ?targetPageId=NNN in query string
    #   4. Numeric segment after wikiId in path
    #      e.g. .../wikis/OneXP.wiki/19165/Digital-Commerce-OPS
    #   5. None → signals "fetch entire wiki from root"

    target_page_id_str = _first_query_value(
        query, "pageId", "page_id", "targetPageId",
    )

    # Look for numeric segment in path after wikiId
    if not target_page_id_str and marker >= 0:
        for part in parts[marker + 2:]:
            if part.isdigit():
                target_page_id_str = part
                logger.debug(
                    "parse_wiki_url: found numeric page ID '%s' in path",
                    target_page_id_str,
                )
                break

    # Validate if present
    if target_page_id_str and not target_page_id_str.isdigit():
        raise ValueError(
            f"Wiki page id '{target_page_id_str}' is not numeric. "
            "Copy the URL directly from your Azure DevOps wiki browser tab."
        )

    # None = no page ID found = fetch entire wiki
    target_page_id = int(target_page_id_str) if target_page_id_str else None

    logger.debug(
        "parse_wiki_url: org=%s project=%s wiki_id=%s page_id=%s",
        organization, project, wiki_id,
        target_page_id if target_page_id is not None else "None (full wiki fetch)",
    )

    return {
        "organization":   organization,
        "project":        project,
        "wiki_id":        wiki_id,
        "target_page_id": target_page_id,   # None = fetch entire wiki
    }

def _fetch_entire_wiki(
    session: requests.Session,
    base:    str,
    org:     str,
    project: str,
    wiki_id: str,
    ver:     str,
) -> list[dict[str, Any]]:
    """
    Fetches ALL pages in the wiki by walking from root with
    recursionLevel=full.

    This is triggered when the input URL is just the wiki root:
      https://dev.azure.com/Org/Project/_wiki/wikis/Wiki.wiki/

    Azure DevOps returns the full tree in one call:
    {
      "path": "/",
      "subPages": [
        {
          "path": "/Digital Commerce OPS",    ← top-level page
          "subPages": [
            { "path": "/Digital Commerce OPS/01_Engagement" },
            { "path": "/Digital Commerce OPS/02_Delivery",
              "subPages": [...]               ← nested at any depth
            }
          ]
        },
        {
          "path": "/Another Top Level Page",
          "subPages": [...]
        }
      ]
    }
    """
    logger.info("[WIKI-ROOT] Fetching entire wiki tree from root...")

    try:
        resp = session.get(
            base,
            params={
                "path":           "/",
                "recursionLevel": "full",       # ← ALL pages at ALL depths
                "includeContent": "false",
                "api-version":    ver,
            },
            timeout=60,
        )
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        logger.error("[WIKI-ROOT] HTTP error: %s", e)
        raise
    except requests.exceptions.RequestException as e:
        logger.error("[WIKI-ROOT] Request error: %s", e)
        raise

    data = _safe_json(resp, "WIKI-ROOT")
    if not data:
        logger.error("[WIKI-ROOT] Empty or invalid response from root tree fetch")
        return []

    # _flatten_tree walks ALL subPages recursively at any depth
    pages = _flatten_tree(data, org, project, wiki_id)

    logger.info("[WIKI-ROOT] ✅ Found %d total wiki page(s)", len(pages))
    _log_page_tree(pages)

    return pages

def _first_query_value(query: dict[str, list[str]], *names: str) -> str:
    for name in names:
        values = query.get(name)
        if values and values[0]:
            return values[0]
    return ""

def _path_to_title(path: str) -> str:
    if not path:
        return ""
    return path.split("/")[-1].replace("-", " ").replace("_", " ").strip()

def _table_row_to_text(row: str, headers: list[str]) -> str:
    cells = [c.strip() for c in row.split("|")[1:-1]]
    if len(cells) == len(headers):
        return ", ".join(f"{headers[j]} = {cells[j]}" for j in range(len(headers)))
    return " | ".join(cells)


def _tables_to_text(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            if len(rows) >= 3 and re.match(r"^[\s|:\-]+$", rows[1]):
                headers = [h.strip() for h in rows[0].split("|")[1:-1]]
                out.append("Table: " + " | ".join(headers))
                for data_row in rows[2:]:
                    converted = _table_row_to_text(data_row, headers)
                    out.append("  - " + converted)
            else:
                out.extend(rows)
        else:
            out.append(line)
            i += 1
    return "\n".join(out)


def _clean_markdown(text: str) -> str:
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = _tables_to_text(text)
    return text.strip()

def _make_page_dict_rest(
    node: dict,
    org: str,
    project: str,
    wiki_id: str,
    depth: int,
) -> dict[str, Any]:
    page_id = node.get("id")
    page_path = node.get("path", "") or ""
    source_label = f"Azure Wiki - {project}/{wiki_id}{page_path or (f'/{page_id}' if page_id is not None else '')}"
    return {
        "id": page_id,
        "path": page_path,
        "title": _path_to_title(page_path),
        "content": "",
        "source": source_label,
        "source_type": "wiki",
        "remote_url": node.get("remoteUrl", "")
        or (f"https://dev.azure.com/{org}/{project}/_wiki/wikis/{wiki_id}/{page_id}"
            if page_id is not None else ""),
        "depth": depth,
        "char_count": 0,
    }

def _flatten_tree(
    node: dict,
    org: str,
    project: str,
    wiki_id: str,
    depth: int = 0,
) -> list[dict[str, Any]]:
    """
    Recursively walks the subPages tree returned by Azure DevOps
    and returns a flat list of ALL wiki navigation child pages.

    Azure DevOps returns the tree like:
    {
      "id": 19165,
      "path": "/Digital Commerce OPS",
      "subPages": [                        ← wiki tree children (NOT hyperlinks)
        {
          "id": ...,
          "path": "/Digital Commerce OPS/01_Engagement Overview",
          "subPages": []
        },
        {
          "id": ...,
          "path": "/Digital Commerce OPS/02_Delivery Model",
          "subPages": [
            {
              "id": ...,
              "path": "/Digital Commerce OPS/02_Delivery Model/SDLC",
              "subPages": [...]            ← grandchildren, any depth
            }
          ]
        }
      ]
    }

    The key parameter is ?recursionLevel=full — without it Azure
    only returns immediate children and stops there.
    """
    result = [_make_page_dict_rest(node, org, project, wiki_id, depth)]
    for child in node.get("subPages", []) or []:
        result.extend(_flatten_tree(child, org, project, wiki_id, depth + 1))
    return result

def _find_node_by_id(node: dict, target_id: Any) -> dict | None:
    """Recursively searches the full wiki tree for a node matching target_id."""
    if node.get("id") == target_id:
        return node
    for child in node.get("subPages", []) or []:
        found = _find_node_by_id(child, target_id)
        if found:
            return found
    return None

def _safe_json(response: requests.Response, label: str) -> dict:
    """
    Guards against Azure returning an HTML login page instead of JSON.

    This happens when:
    - The PAT is expired or missing
    - The PAT lacks Wiki (Read) scope
    - The request is missing the X-TFS-FedAuthRedirect: Suppress header

    Without this guard, response.json() raises JSONDecodeError and
    crashes the entire pipeline.
    """
    ctype = response.headers.get("Content-Type", "")
    if "text/html" in ctype:
        logger.error(
            "[%s] Got HTML instead of JSON (auth issue?) "
            "status=%s body=%s",
            label, response.status_code, response.text[:200],
        )
        return {}
    try:
        return response.json()
    except Exception as exc:
        logger.error(
            "[%s] JSON decode failed: %s | body=%s",
            label, exc, response.text[:200],
        )
        return {}

def _make_session(pat: str) -> requests.Session:
    """
    Creates a PAT-authenticated session with headers that prevent
    Azure DevOps from redirecting to the login page (status 203 + HTML).

    Without X-TFS-FedAuthRedirect: Suppress, Azure returns:
      Status : 203
      Content-Type: text/html
      Body   : Azure DevOps Sign In page   ← causes JSONDecodeError

    With it, Azure uses PAT auth directly:
      Status : 200
      Content-Type: application/json       ← correct ✅
    """
    session = requests.Session()
    session.auth = HTTPBasicAuth("", pat)
    session.headers.update({
        # Stops Azure from redirecting unauthenticated requests
        # to the browser login page instead of returning 401/JSON.
        "X-TFS-FedAuthRedirect":     "Suppress",
        "X-VSS-ForceMsaPassThrough": "true",
    })
    return session

def _discover_pages(
    session: requests.Session,
    base: str,
    target_page_id: int,
    org: str,
    project: str,
    wiki_id: str,
    ver: str,
) -> list[dict[str, Any]]:
    """
    Discovers ALL wiki navigation child pages under the target page.

    Returns subPages tree children — NOT hyperlinks found inside
    page content. Uses ?recursionLevel=full to get the entire
    subtree in a single API call.

    Tries 2 strategies in order:

    ┌─────────────────────────────────────────────────────────────┐
    │ Strategy 1 — Fetch by page ID → get exact path → tree fetch │
    │                                                             │
    │   GET /pages/{id}                                           │
    │     → exact path (e.g. "/Digital Commerce OPS")            │
    │   GET /pages?path={exact}&recursionLevel=full               │
    │     → full subPages tree rooted at that page               │
    │                                                             │
    │ Strategy 2 — Root tree search by page ID                    │
    │                                                             │
    │   GET /pages?path=/&recursionLevel=full                     │
    │     → entire wiki tree                                      │
    │   _find_node_by_id() → extract subtree at target node      │
    └─────────────────────────────────────────────────────────────┘
    """

    # ── Strategy 1: Fetch exact path by page ID ───────────────────
    # Most reliable — page ID never changes regardless of renames.
    logger.debug("[DISCOVER] Strategy 1: fetch exact path for page ID %d", target_page_id)
    try:
        id_resp = session.get(
            f"{base}/{target_page_id}",
            params={"includeContent": "false", "api-version": ver},
            timeout=30,
        )
        id_resp.raise_for_status()
        id_data    = _safe_json(id_resp, "DISCOVER-ID")
        exact_path = id_data.get("path", "") if id_data else ""

        if exact_path:
            logger.debug("[DISCOVER] Strategy 1: exact path = '%s'", exact_path)

            tree_resp = session.get(
                base,
                params={
                    "path":           exact_path,
                    "recursionLevel": "full",     # ← returns ALL child pages at any depth
                    "includeContent": "false",
                    "api-version":    ver,
                },
                timeout=60,
            )
            tree_resp.raise_for_status()
            tree_data = _safe_json(tree_resp, "DISCOVER-TREE")

            if tree_data:
                pages = _flatten_tree(tree_data, org, project, wiki_id)
                if pages:
                    logger.info(
                        "[DISCOVER] ✅ Strategy 1 succeeded — %d child page(s) "
                        "under '%s'",
                        len(pages), exact_path,
                    )
                    _log_page_tree(pages)
                    return pages

    except Exception as exc:
        logger.warning("[DISCOVER] Strategy 1 failed: %s", exc)

    # ── Strategy 2: Fetch entire root tree, find node by ID ───────
    # Fallback when the /pages/{id} endpoint returns 404 or empty path.
    logger.debug("[DISCOVER] Strategy 2: search root tree for page ID %d", target_page_id)
    try:
        root_resp = session.get(
            base,
            params={
                "path":           "/",
                "recursionLevel": "full",         # ← full tree from root
                "includeContent": "false",
                "api-version":    ver,
            },
            timeout=60,
        )
        root_resp.raise_for_status()
        root_data = _safe_json(root_resp, "DISCOVER-ROOT")

        if root_data:
            target_node = _find_node_by_id(root_data, target_page_id)
            if target_node:
                pages = _flatten_tree(target_node, org, project, wiki_id)
                logger.info(
                    "[DISCOVER] ✅ Strategy 2 succeeded — %d child page(s) "
                    "under ID %d",
                    len(pages), target_page_id,
                )
                _log_page_tree(pages)
                return pages
            else:
                logger.warning(
                    "[DISCOVER] Strategy 2: page ID %d not found in root tree",
                    target_page_id,
                )

    except Exception as exc:
        logger.warning("[DISCOVER] Strategy 2 failed: %s", exc)

    logger.error(
        "[DISCOVER] ❌ All strategies failed for page ID %d. "
        "Check PAT scopes (Wiki Read) and wiki_id.",
        target_page_id,
    )
    return []

def _log_page_tree(pages: list[dict[str, Any]]) -> None:
    """Logs the discovered wiki child page tree structure at DEBUG level."""
    logger.debug("[DISCOVER] Wiki child page tree:")
    for page in pages:
        indent = "  " * page.get("depth", 0)
        icon   = "📁" if any(
            p.get("depth", 0) > page.get("depth", 0)
            and p["path"].startswith(page["path"])
            for p in pages
        ) else "📄"
        logger.debug(
            "  %s%s [ID:%s] %s",
            indent, icon, page.get("id"), page.get("title"),
        )


def print_page_tree(pages: list[dict[str, Any]]) -> None:
    """Prints the page tree to terminal with emoji icons."""
    if not pages:
        print("  (no pages)")
        return

    # If all depths are 0 (HTML scraper), infer depth from URL path
    has_real_depth = any(p.get("depth", 0) > 0 for p in pages)

    print("\n📚 Page Tree:")
    print("=" * 60)
    for page in pages:
        if has_real_depth:
            depth = page.get("depth", 0)
        else:
            path = page.get("path", "")
            depth = path.rstrip("/").count("/") - 1 if path else 0
            if depth < 0:
                depth = 0

        indent = "  " * depth
        is_container = any(
            p.get("path", "").startswith(page.get("path", "") + "/")
            for p in pages
            if p.get("path") != page.get("path")
        )
        icon = "📁" if is_container else "📄"
        title = page.get("title", page.get("path", "?"))
        page_id = page.get("id")
        suffix = f" [ID:{page_id}]" if page_id is not None else ""
        connector = "└─ " if depth == 0 else ""
        print(f"{indent}{connector}{icon} {title}{suffix}")

    print("=" * 60)
    print(f"Total: {len(pages)} page(s)\n")

def _fetch_via_rest_api(
    pat: str,
    wiki_url: str,
) -> list[dict[str, Any]]:
    """Fetch Azure DevOps wiki pages via the REST API."""
    params         = parse_wiki_url(wiki_url)
    org            = str(params["organization"])
    proj           = str(params["project"])
    wiki_id        = str(params["wiki_id"])

    # ── FIX: don't cast to int — it can be None (root wiki URL) ───
    # ❌ OLD: target_page_id = int(params["target_page_id"])
    # ✅ NEW: preserve None so the root-fetch branch triggers
    target_page_id = params["target_page_id"]   # int | None

    ver     = os.getenv("AZURE_API_VERSION", "7.1")
    base    = f"https://dev.azure.com/{org}/{proj}/_apis/wiki/wikis/{wiki_id}/pages"
    session = _make_session(pat)

    # ── Route: full wiki vs specific page ─────────────────────────
    if target_page_id is None:
        logger.info(
            "[WIKI] No page ID found — fetching entire wiki from root. "
            "wiki_id=%s", wiki_id,
        )
        pages = _fetch_entire_wiki(session, base, org, proj, wiki_id, ver)
    else:
        logger.info(
            "[WIKI] Page ID=%d found — fetching page and all child pages.",
            target_page_id,
        )
        pages = _discover_pages(
            session, base, target_page_id, org, proj, wiki_id, ver,
        )

    if not pages:
        raise ValueError(
            f"No wiki pages found. "
            f"Check the URL and that the PAT has Wiki (Read) access. "
            f"wiki_id={wiki_id}"
        )

    def _hydrate(page: dict[str, Any]) -> dict[str, Any]:
        """Fetch full Markdown content for a single page."""
        content = ""
        try:
            r = session.get(
                base,
                params={
                    "path":           page["path"],
                    "includeContent": "true",
                    "api-version":    ver,
                },
                timeout=30,
            )
            r.raise_for_status()
            data    = _safe_json(r, f"HYDRATE:{page['title']}")
            content = data.get("content", "") or "" if data else ""
        except Exception:
            page_id = page.get("id")
            if page_id is not None:
                try:
                    r = session.get(
                        f"{base}/{page_id}",
                        params={"includeContent": "true", "api-version": ver},
                        timeout=30,
                    )
                    r.raise_for_status()
                    data    = _safe_json(r, f"HYDRATE-ID:{page['title']}")
                    content = data.get("content", "") or "" if data else ""
                except Exception:
                    logger.warning(
                        "Failed to fetch content for wiki page id=%s (%s)",
                        page_id, page.get("path"),
                    )
        clean              = _clean_markdown(content)
        page["content"]    = clean
        page["char_count"] = len(clean)
        return page

    with ThreadPoolExecutor(max_workers=8) as pool:
        pages = list(pool.map(_hydrate, pages))

    with_content = sum(1 for p in pages if p.get("content"))
    logger.info(
        "Fetched %d wiki page(s); %d have content, %d empty.",
        len(pages), with_content, len(pages) - with_content,
    )
    return pages

# ──────────────────────────────────────────────
#  Generic HTML scraper (non-Azure-DevOps URLs)
# ──────────────────────────────────────────────

def _get_child_links(session: requests.Session, url: str) -> list[str]:
    """Fetch the parent page and extract all child page links (same domain + child path)."""
    from bs4 import BeautifulSoup

    response = session.get(url, timeout=30)
    response.raise_for_status()

    soup        = BeautifulSoup(response.text, "html.parser")
    parsed_base = urlparse(url)
    base_path   = parsed_base.path
    child_links: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        href         = anchor["href"]
        absolute_url = urljoin(url, href)
        parsed_url   = urlparse(absolute_url)

        is_same_domain = parsed_url.netloc == parsed_base.netloc
        is_child_path  = (
            parsed_url.path.startswith(base_path)
            and parsed_url.path != base_path
        )
        is_http = parsed_url.scheme in ("http", "https")

        if is_same_domain and is_child_path and is_http:
            clean_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
            child_links.add(clean_url)

    return list(child_links)

def _fetch_page(session: requests.Session, url: str) -> dict[str, Any]:
    """Fetch a single page and return a normalized page dict."""
    from bs4 import BeautifulSoup

    try:
        response = session.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Failed to fetch %s: %s", url, e)
        return {
            "id":          url,
            "path":        urlparse(url).path,
            "title":       urlparse(url).path.split("/")[-1] or url,
            "content":     "",
            "source":      url,
            "source_type": "web",
            "remote_url":  url,
            "depth":       0,
            "char_count":  0,
        }

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "button"]):
        tag.decompose()

    main_content = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", class_="wiki-content")
        or soup.find("div", class_="content")
        or soup.find("div", id="content")
        or soup.find("body")
    )

    raw_text   = main_content.get_text(separator="\n", strip=True) if main_content else ""
    lines      = [line.strip() for line in raw_text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)

    title_tag  = soup.find("title")
    page_title = title_tag.get_text(strip=True) if title_tag else url

    return {
        "id":          url,
        "path":        urlparse(url).path,
        "title":       page_title,
        "content":     clean_text,
        "source":      url,
        "source_type": "web",
        "remote_url":  url,
        "depth":       0,
        "char_count":  len(clean_text),
    }

def _fetch_via_html_scraping(
    pat: str,
    wiki_url: str,
) -> list[dict[str, Any]]:
    """Fetch a URL and all its child pages via HTML scraping."""
    session = requests.Session()
    if pat:
        session.auth = HTTPBasicAuth("", pat)

    child_urls = _get_child_links(session, wiki_url)
    all_urls   = [wiki_url] + child_urls

    logger.info("Found %d child page(s) under %s", len(child_urls), wiki_url)

    with ThreadPoolExecutor(max_workers=8) as pool:
        pages = list(pool.map(lambda url: _fetch_page(session, url), all_urls))

    with_content = sum(1 for p in pages if p.get("content"))
    logger.info(
        "Fetched %d page(s); %d have content, %d empty.",
        len(pages), with_content, len(pages) - with_content,
    )
    return pages

# ──────────────────────────────────────────────
#  Public entry point
# ──────────────────────────────────────────────

def fetch_wiki_pages_from_url(
    pat: str = "",
    wiki_url: str = "",
) -> list[dict[str, Any]]:
    """Fetch a URL and its child pages.

    For Azure DevOps wiki URLs (dev.azure.com, *.visualstudio.com) the Azure
    DevOps REST API is used, which returns structured wiki markdown content.

    For all other URLs, generic HTML scraping is used: the parent page is
    fetched, child links are discovered (same domain + path prefix), and each
    child page's text content is extracted.
    """
    parsed = urlparse(wiki_url)
    host   = parsed.netloc.lower()

    if host == "dev.azure.com" or host.endswith(".visualstudio.com"):
        pages = _fetch_via_rest_api(pat=pat, wiki_url=wiki_url)
    else:
        pages = _fetch_via_html_scraping(pat=pat, wiki_url=wiki_url)

    print_page_tree(pages)
    return pages