"""USPTO Open Data Portal (ODP) bulk-data API client.

Endpoint reference (verified 2026-04-27):
    Auth: header  X-API-KEY: <key>
    Get a key:    https://data.uspto.gov  →  My ODP  →  My API Key
    Search:       GET /api/v1/datasets/products/search
    Get product:  GET /api/v1/datasets/products/{productIdentifier}
    Download:     GET /api/v1/datasets/products/files/{productIdentifier}/{fileName}

Response shape (search and product detail):
    { "count": N, "bulkDataProductBag": [ {productIdentifier, productTitleText,
        productFromDate, productToDate, productTotalFileSize,
        productFileBag: { count, fileDataBag: [ {fileName, fileSize,
            fileDataFromDate, fileDataToDate, fileDownloadURI, ...} ] }
      } ] }

The legacy hosts bulkdata.uspto.gov and trademarks.reedtech.com are retired.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import requests

ODP_BASE = "https://api.uspto.gov/api/v1"

PRODUCT_TRADEMARK_ANNUAL_APPLICATIONS = "TRTYRAP"


def _key() -> str:
    k = os.environ.get("USPTO_ODP_API_KEY")
    if not k:
        raise RuntimeError(
            "USPTO_ODP_API_KEY not set. Get a key at "
            "https://data.uspto.gov (My ODP > My API Key) and put it in .env"
        )
    return k


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"X-API-KEY": _key(), "Accept": "application/json"})
    return s


@dataclass
class FileEntry:
    file_name: str
    download_url: str
    size_bytes: int
    data_from_date: str | None
    data_to_date: str | None


def get_product(product_id: str) -> dict:
    r = _session().get(f"{ODP_BASE}/datasets/products/{product_id}", timeout=60)
    r.raise_for_status()
    body = r.json()
    bag = body.get("bulkDataProductBag", [])
    if not bag:
        raise RuntimeError(f"product {product_id} not found")
    return bag[0]


def list_product_files(product_id: str) -> list[FileEntry]:
    p = get_product(product_id)
    files = (p.get("productFileBag") or {}).get("fileDataBag") or []
    return [
        FileEntry(
            file_name=f["fileName"],
            download_url=f["fileDownloadURI"],
            size_bytes=int(f.get("fileSize") or 0),
            data_from_date=f.get("fileDataFromDate"),
            data_to_date=f.get("fileDataToDate"),
        )
        for f in files
    ]


def latest_backfile_files(product_id: str) -> list[FileEntry]:
    """Return the parts of the most recent backfile (highest fileDataToDate)."""
    files = list_product_files(product_id)
    if not files:
        raise RuntimeError(f"product {product_id} has no files")
    latest = max(f.data_to_date or "" for f in files)
    parts = [f for f in files if f.data_to_date == latest]
    parts.sort(key=lambda f: f.file_name)
    return parts


def stream_download(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with _session().get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with dest.open("wb") as fh:
            for buf in r.iter_content(chunk_size=chunk):
                if buf:
                    fh.write(buf)
    return dest
