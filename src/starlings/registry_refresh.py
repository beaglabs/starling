"""Explicit public NARA metadata refresh. Never runs during inference or training."""

from __future__ import annotations

import hashlib
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

from .datasets import write_json
from .schemas import Authority, Category, Registry

SOURCE = "https://www.archives.gov/cui/registry/category-list"


def fetch(url):
    if urlparse(url).hostname != "www.archives.gov":
        raise ValueError("Registry refresh only fetches www.archives.gov")
    last = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "Starlings/0.1 public-registry-refresh"}
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                if urlparse(response.url).hostname != "www.archives.gov":
                    raise ValueError("Unexpected registry redirect")
                return response.read(4_000_000)
        except Exception as exc:
            last = exc
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError(f"Registry source failed: {url}: {last}")


def refresh(output, workers=4):
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError("Install the registry extra: pip install '.[registry]'") from exc
    soup = BeautifulSoup(fetch(SOURCE), "html.parser")
    entries = []
    for row in soup.select("table tr"):
        cells = row.find_all("td", recursive=False)
        if len(cells) < 2:
            continue
        group = cells[0].get_text(" ", strip=True)
        for link in cells[1].select('a[href*="category-detail"]'):
            entries.append((group, link.get_text(" ", strip=True), urljoin(SOURCE, link["href"])))
    if not entries or len({url for _, _, url in entries}) != len(entries):
        raise ValueError(
            "Registry index structure changed or categories duplicated; refusing partial refresh"
        )

    def parse(entry):
        group, name, url = entry
        html = fetch(url)
        detail = BeautifulSoup(html, "html.parser")
        description, authorities, authority_refs = None, [], []
        category_marking, authority_details = "", []
        for row in detail.select("table tr"):
            cells = row.find_all(["td", "th"], recursive=False)
            if len(cells) < 2:
                continue
            label = cells[0].get_text(" ", strip=True).lower()
            if "category description" in label:
                description = cells[1].get_text(" ", strip=True)
            if "category marking" in label:
                category_marking = cells[1].get_text(" ", strip=True)
        # Authorities are data rows beneath a multi-column header, not the header itself.
        for table in detail.select("table"):
            rows = table.find_all("tr")
            if not rows:
                continue
            header_index = next(
                (
                    i
                    for i, row in enumerate(rows)
                    if "safeguarding and/or dissemination authority"
                    in row.get_text(" ", strip=True).lower()
                ),
                None,
            )
            if header_index is None:
                continue
            for row in rows[header_index + 1 :]:
                cells = row.find_all(["td", "th"], recursive=False)
                if not cells:
                    continue
                text = cells[0].get_text(" ", strip=True)
                if text:
                    values = [cell.get_text(" ", strip=True) for cell in cells]
                    authority_details.append(
                        Authority(
                            citation=text,
                            control_type=values[1] if len(values) > 1 else "",
                            banner_marking=values[2] if len(values) > 2 else "",
                            sanctions=values[3] if len(values) > 3 else "",
                            refs=[urljoin(url, a["href"]) for a in cells[0].select("a[href]")],
                        )
                    )
                    authorities.append(text)
                    authority_refs += [urljoin(url, a["href"]) for a in cells[0].select("a[href]")]
        if not description:
            raise ValueError(f"Missing category description: {url}; refusing partial refresh")
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        return Category(
            id=slug,
            name=name,
            group=group,
            description=description,
            category_marking=category_marking,
            authority_details=authority_details,
            authorities=authorities,
            authority_refs=sorted(set(authority_refs)),
            source_url=url,
            source_sha256=hashlib.sha256(html).hexdigest(),
        )

    with ThreadPoolExecutor(max_workers=max(1, min(workers, 8))) as pool:
        categories = list(pool.map(parse, entries))
    date = datetime.now(timezone.utc)
    registry = Registry(
        version=f"nara-cui-{date.date()}",
        retrieved_at=date.isoformat(),
        source_url=SOURCE,
        categories=categories,
    )
    write_json(output, registry.model_dump())
    return {"version": registry.version, "categories": len(categories), "output": str(output)}
