import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import requests
from requests.auth import HTTPBasicAuth

logger = logging.getLogger(__name__)

AZURE_DEVOPS_ORG = os.getenv("AZURE_DEVOPS_ORG", "")
AZURE_DEVOPS_PROJECT = os.getenv("AZURE_DEVOPS_PROJECT", "")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "7.1")


def parse_wiki_url(wiki_url: str) -> dict[str, str | int]:
    """Extract Azure DevOps wiki fetch parameters from a browser URL."""
    parsed = urlparse(wiki_url.strip())
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Enter a complete Azure DevOps wiki URL.")

    host = parsed.netloc.lower()
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    query = parse_qs(parsed.query)

    organization = ""
    project = ""
    if host == "dev.azure.com":
        if len(parts) < 2:
            raise ValueError("Azure DevOps URL must include organization and project.")
        organization = parts[0]
        project = parts[1]
    elif host.endswith(".visualstudio.com"):
        organization = parsed.netloc.split(".")[0]
        if not parts:
            raise ValueError("Azure DevOps URL must include a project.")
        project = parts[0]
    else:
        raise ValueError("URL must be from dev.azure.com or visualstudio.com.")

    wiki_id = _first_query_value(query, "wikiIdentifier", "wiki_id", "wikiId")
    target_page_id = _first_query_value(query, "pageId", "page_id", "targetPageId")

    try:
        marker = parts.index("wikis")
    except ValueError:
        marker = -1

    if marker >= 0 and len(parts) > marker + 1:
        wiki_id = wiki_id or parts[marker + 1]
        if len(parts) > marker + 2 and parts[marker + 2].isdigit():
            target_page_id = target_page_id or parts[marker + 2]

    if not wiki_id:
        raise ValueError("Could not find wiki id in the URL.")
    if not target_page_id or not str(target_page_id).isdigit():
        raise ValueError("Could not find a numeric wiki page id in the URL.")

    return {
        "organization": organization,
        "project": project,
        "wiki_id": wiki_id,
        "target_page_id": int(target_page_id),
    }


def _first_query_value(query: dict[str, list[str]], *names: str) -> str:
    for name in names:
        values = query.get(name)
        if values and values[0]:
            return values[0]
    return ""


def _path_to_title(path: str) -> str:
    """Derive a human-readable title from a wiki page path."""
    if not path:
        return ""
    return path.split("/")[-1].replace("-", " ").replace("_", " ").strip()


def _clean_markdown(text: str) -> str:
    """Strip images, raw HTML, comments and excess whitespace from wiki markdown."""
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)        # image embeds
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)  # HTML comments
    text = re.sub(r"<[^>]+>", "", text)                # raw HTML tags
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)  # horizontal rules
    text = re.sub(r"\n{3,}", "\n\n", text)             # collapse blank lines
    return text.strip()


def _make_page_dict(node: dict, org: str, project: str, wiki_id: str, depth: int) -> dict[str, Any]:
    """Build our normalized page record (without content) from an API tree node."""
    page_id = node.get("id")
    page_path = node.get("path", "") or ""
    return {
        "id": page_id,
        "path": page_path,
        "title": _path_to_title(page_path),
        "content": "",
        "source": f"Azure Wiki - {project}/{wiki_id}{page_path or f'/{page_id}'}",
        "source_type": "wiki",
        "remote_url": node.get("remoteUrl", "")
        or f"https://dev.azure.com/{org}/{project}/_wiki/wikis/{wiki_id}/{page_id}",
        "depth": depth,
        "char_count": 0,
    }


def _flatten_tree(node: dict, org: str, project: str, wiki_id: str, depth: int = 0) -> list[dict[str, Any]]:
    """Flatten an Azure DevOps wiki page tree into a depth-ordered list.

    The API nests children under ``subPages`` (not ``childPages``); reading the
    wrong key is why earlier traversals only ever found the root page.
    """
    result = [_make_page_dict(node, org, project, wiki_id, depth)]
    for child in node.get("subPages", []) or []:
        result.extend(_flatten_tree(child, org, project, wiki_id, depth + 1))
    return result


def _find_node_by_id(node: dict, target_id: Any) -> dict | None:
    if node.get("id") == target_id:
        return node
    for child in node.get("subPages", []) or []:
        found = _find_node_by_id(child, target_id)
        if found:
            return found
    return None


def _discover_pages(
    session: requests.Session,
    base: str,
    target_page_id: int,
    org: str,
    project: str,
    wiki_id: str,
    ver: str,
) -> list[dict[str, Any]]:
    """Discover the target page and all descendants, trying several strategies.

    1. Resolve the target's exact path by id, then pull the subtree by path.
    2. Fall back to fetching the whole wiki tree and slicing out the subtree
       rooted at the target id.
    """
    # Strategy 1: exact path by id, then subtree by path.
    try:
        r = session.get(
            f"{base}/{target_page_id}",
            params={"includeContent": "false", "api-version": ver},
            timeout=30,
        )
        r.raise_for_status()
        exact_path = r.json().get("path", "")
        if exact_path:
            tr = session.get(
                f"{base}",
                params={
                    "path": exact_path,
                    "recursionLevel": "full",
                    "includeContent": "false",
                    "api-version": ver,
                },
                timeout=60,
            )
            tr.raise_for_status()
            pages = _flatten_tree(tr.json(), org, project, wiki_id)
            if pages:
                logger.info("Wiki discovery: strategy 1 (by path) found %d page(s)", len(pages))
                return pages
    except Exception as exc:
        logger.warning("Wiki discovery strategy 1 failed: %s", exc)

    # Strategy 2: fetch the whole tree, locate the target node by id.
    try:
        r = session.get(
            f"{base}",
            params={
                "path": "/",
                "recursionLevel": "full",
                "includeContent": "false",
                "api-version": ver,
            },
            timeout=60,
        )
        r.raise_for_status()
        node = _find_node_by_id(r.json(), target_page_id)
        if node:
            pages = _flatten_tree(node, org, project, wiki_id)
            logger.info("Wiki discovery: strategy 2 (root tree) found %d page(s)", len(pages))
            return pages
    except Exception as exc:
        logger.warning("Wiki discovery strategy 2 failed: %s", exc)

    return []


def fetch_wiki_pages(
    pat: str,
    wiki_id: str,
    target_page_id: int,
    organization: str | None = None,
    project: str | None = None,
    api_version: str | None = None,
) -> list[dict[str, Any]]:
    org = organization or AZURE_DEVOPS_ORG
    proj = project or AZURE_DEVOPS_PROJECT
    ver = api_version or AZURE_API_VERSION

    if not org or not proj:
        raise ValueError(
            "Azure DevOps organization and project are required. "
            "Pass them as arguments or set AZURE_DEVOPS_ORG / AZURE_DEVOPS_PROJECT."
        )

    base = f"https://dev.azure.com/{org}/{proj}/_apis/wiki/wikis/{wiki_id}/pages"
    session = requests.Session()
    session.auth = HTTPBasicAuth("", pat)

    # 1. Discover the page tree (ids + paths only — tree calls don't return content).
    pages = _discover_pages(session, base, target_page_id, org, proj, wiki_id, ver)
    if not pages:
        raise ValueError(
            f"No wiki pages found for page id {target_page_id}. "
            "Check the URL and that the PAT has Wiki (Read) access."
        )

    # 2. Fetch each page's markdown individually (path first, id as fallback),
    #    in parallel, then clean it. This is what actually pulls every child page.
    def _hydrate(page: dict[str, Any]) -> dict[str, Any]:
        content = ""
        try:
            r = session.get(
                f"{base}",
                params={"path": page["path"], "includeContent": "true", "api-version": ver},
                timeout=30,
            )
            r.raise_for_status()
            content = r.json().get("content", "") or ""
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
                    content = r.json().get("content", "") or ""
                except Exception:
                    logger.warning(
                        "Failed to fetch content for wiki page id=%s (%s)", page_id, page.get("path")
                    )
        clean = _clean_markdown(content)
        page["content"] = clean
        page["char_count"] = len(clean)
        return page

    with ThreadPoolExecutor(max_workers=8) as pool:
        pages = list(pool.map(_hydrate, pages))

    with_content = sum(1 for p in pages if p.get("content"))
    logger.info(
        "Fetched %d wiki page(s); %d have content, %d empty.",
        len(pages),
        with_content,
        len(pages) - with_content,
    )
    return pages


def fetch_wiki_pages_from_url(
    pat: str,
    wiki_url: str,
    api_version: str | None = None,
) -> list[dict[str, Any]]:
    params = parse_wiki_url(wiki_url)
    return fetch_wiki_pages(
        pat=pat,
        wiki_id=str(params["wiki_id"]),
        target_page_id=int(params["target_page_id"]),
        organization=str(params["organization"]),
        project=str(params["project"]),
        api_version=api_version,
    )
