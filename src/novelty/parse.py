"""Streaming parser for USPTO Trademark Case File XML.

DTD reference: U.S. Trademark Applications v2.0
Records are <case-file> elements; we iterparse to keep memory bounded.
"""

from __future__ import annotations

import gzip
import io
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree


@dataclass
class CaseFile:
    serial_number: str
    filing_date: str | None
    registration_date: str | None
    status_code: str | None
    mark_identification: str | None
    nice_classes: list[str] = field(default_factory=list)
    us_classes: list[str] = field(default_factory=list)
    owner_name: str | None = None
    owner_state: str | None = None  # US state (2-char) if US-domiciled
    owner_country: str | None = None  # 2-char ISO country (e.g., 'US', 'CN', 'JP')
    goods_services: str = ""


def _text(elem, path: str) -> str | None:
    n = elem.find(path)
    if n is None or n.text is None:
        return None
    return n.text.strip() or None


def _record_to_case_file(elem) -> CaseFile:
    serial = _text(elem, "serial-number") or ""

    header = elem.find("case-file-header")
    filing = registration = status = mark = None
    if header is not None:
        filing = _text(header, "filing-date")
        registration = _text(header, "registration-date")
        status = _text(header, "status-code")
        mark = _text(header, "mark-identification")

    nice: list[str] = []
    us: list[str] = []
    classifications = elem.find("classifications")
    if classifications is not None:
        for c in classifications.findall("classification"):
            for n in c.findall("international-code"):
                if n.text and n.text.strip():
                    nice.append(n.text.strip())
            for n in c.findall("us-code"):
                if n.text and n.text.strip():
                    us.append(n.text.strip())

    owner_name = None
    owner_state = None
    owner_country = None
    owners = elem.find("case-file-owners")
    if owners is not None:
        first = owners.find("case-file-owner")
        if first is not None:
            owner_name = _text(first, "party-name")
            # USPTO Trademark Case File DTD: case-file-owner has child
            # elements address-1, address-2, city, state (US 2-letter),
            # postcode, country (2-letter ISO). Older records may omit.
            owner_state = _text(first, "state")
            owner_country = _text(first, "country")

    parts: list[str] = []
    statements = elem.find("case-file-statements")
    if statements is not None:
        for s in statements.findall("case-file-statement"):
            type_code = _text(s, "type-code") or ""
            if type_code.startswith("GS"):
                txt = _text(s, "text")
                if txt:
                    parts.append(txt)
    goods = " ".join(parts)

    return CaseFile(
        serial_number=serial,
        filing_date=filing,
        registration_date=registration,
        status_code=status,
        mark_identification=mark,
        nice_classes=sorted(set(nice)),
        us_classes=sorted(set(us)),
        owner_name=owner_name,
        owner_state=owner_state,
        owner_country=owner_country,
        goods_services=goods,
    )


def iter_case_files(xml_stream) -> Iterator[CaseFile]:
    """Stream <case-file> records from an XML byte stream."""
    context = etree.iterparse(xml_stream, events=("end",), tag="case-file", recover=True)
    for _, elem in context:
        try:
            yield _record_to_case_file(elem)
        finally:
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]


def iter_zip_xml(zip_path: Path) -> Iterator[CaseFile]:
    """Stream case files from a USPTO trademark annual ZIP without unpacking to disk."""
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if not (member.endswith(".xml") or member.endswith(".xml.gz")):
                continue
            with zf.open(member) as fh:
                if member.endswith(".gz"):
                    fh = gzip.GzipFile(fileobj=io.BufferedReader(fh))
                yield from iter_case_files(fh)
