import logging
import os
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


def _flatten_pages(page: dict, org: str, project: str, wiki_id: str, depth: int = 0) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    page_id = page.get("id")
    page_path = page.get("path", "")
    segments = [s for s in page_path.split("/") if s]
    title = segments[-1] if segments else ""
    content = page.get("content", "")
    source = f"Azure Wiki - {project}/{wiki_id}{page_path or f'/{page_id}'}"

    result.append({
        "id": page_id,
        "path": page_path,
        "title": title,
        "content": content,
        "source": source,
        "source_type": "text",
        "remote_url": f"https://dev.azure.com/{org}/{project}/_wiki/wikis/{wiki_id}/{page_id}",
        "depth": depth,
        "char_count": len(content),
    })

    for child in page.get("childPages", []):
        result.extend(_flatten_pages(child, org, project, wiki_id, depth + 1))

    return result


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

    url = (
        f"https://dev.azure.com/{org}/{proj}/_apis/wiki/wikis/{wiki_id}"
        f"/pages/{target_page_id}"
    )
    params = {
        "includeContent": "true",
        "recursionLevel": "full",
        "api-version": ver,
    }

    logger.info("Fetching wiki pages from %s ...", url)
    resp = requests.get(url, params=params, auth=HTTPBasicAuth("", pat), timeout=60)
    resp.raise_for_status()
    root = resp.json()

    return _flatten_pages(root, org, proj, wiki_id)


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
