"""Pull the fields we need out of one Rechtspraak XML file.

A downloaded file mixes metadata (court, date, labels) with the ruling text.
This module is the only place that knows how those XML tags map onto the
columns the rest of the pipeline uses. If Rechtspraak renames a field, this
is the file to fix -- and because the XML is already on disk, the fix can be
re-run without downloading anything again.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


def _local_name(tag: str) -> str:
    """Strip the XML namespace, leaving the short tag name.

    ElementTree stores tags as ``{http://purl.org/dc/terms/}subject``.
    Comparing on the short name keeps the parser readable and stops it
    breaking if a prefix (``dcterms:`` vs the full URL) is written differently.
    """
    if tag.startswith("{") and "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _texts_with_name(root: ET.Element, name: str) -> list[str]:
    """Return every non-empty text value of tags with this short name."""
    values: list[str] = []
    for element in root.iter():
        if _local_name(element.tag) != name:
            continue
        value = (element.text or "").strip()
        if value:
            values.append(value)
    return values


def _first_text(root: ET.Element, name: str) -> str:
    values = _texts_with_name(root, name)
    return values[0] if values else ""


def _block_text(root: ET.Element, name: str) -> str:
    """Join all visible text inside the first block with this name.

    ``<uitspraak>`` holds the ruling as a tree of ``<para>`` tags. Walking
    every text node keeps section numbers and quoted passages, and does not
    depend on the exact nesting (``parablock``, ``paragroup``, ...).
    """
    for element in root.iter():
        if _local_name(element.tag) != name:
            continue
        parts = [chunk.strip() for chunk in element.itertext() if chunk.strip()]
        if parts:
            return "\n".join(parts)
    return ""


def expand_rechtsgebied(raw: str, composite_to_pair: dict) -> list[str]:
    """Turn one official tag into the labels we actually train on.

    Rechtspraak writes a sub-area as ``Bestuursrecht; Vreemdelingenrecht``.
    The 2024 trainer treated that whole string as one class, so the parent
    area and the child area never helped each other. Here the string is
    split into ``Bestuursrecht`` and ``Vreemdelingenrecht``.

    Unknown values (not in the official list) are split on ``;`` the same
    way, so a newly added area still becomes usable labels instead of being
    dropped on the floor.
    """
    raw = " ".join(raw.split())
    if not raw:
        return []

    pair = composite_to_pair.get(raw)
    if pair is not None:
        return [part for part in pair if part]

    return [part.strip() for part in raw.split(";") if part.strip()]


def parse_xml(
    xml_text: str,
    composite_to_pair: dict,
) -> dict:
    """Parse one ruling. Returns a dict; never raises on a single bad field."""
    root = ET.fromstring(xml_text)

    ecli = _first_text(root, "identifier")
    # The HTML description repeats an identifier that is a URL, not an ECLI.
    if not ecli.startswith("ECLI:"):
        for value in _texts_with_name(root, "identifier"):
            if value.startswith("ECLI:"):
                ecli = value
                break

    raw_subjects = _texts_with_name(root, "subject")
    rechtsgebieden: list[str] = []
    seen_rg: set[str] = set()
    for raw in raw_subjects:
        for label in expand_rechtsgebied(raw, composite_to_pair):
            if label not in seen_rg:
                seen_rg.add(label)
                rechtsgebieden.append(label)

    # XML tag is still `procedure`; the public name is bijzondere kenmerken.
    bijzondere_kenmerken: list[str] = []
    seen_bk: set[str] = set()
    for raw in _texts_with_name(root, "procedure"):
        name = " ".join(raw.split())
        if name and name not in seen_bk:
            seen_bk.add(name)
            bijzondere_kenmerken.append(name)

    # Prefer the full ruling. A few files are opinions (conclusie) instead.
    # The short summary is a last resort and is usually too brief to keep.
    text = _block_text(root, "uitspraak")
    text_source = "uitspraak"
    if not text:
        text = _block_text(root, "conclusie")
        text_source = "conclusie"
    if not text:
        text = _block_text(root, "inhoudsindicatie")
        text_source = "inhoudsindicatie"

    return {
        "ecli": ecli,
        "court": _first_text(root, "creator"),
        "date": _first_text(root, "date"),
        "text": text,
        "text_length": len(text),
        "text_source": text_source if text else "",
        "rechtsgebieden": rechtsgebieden,
        "rechtsgebieden_raw": raw_subjects,
        "bijzondere_kenmerken": bijzondere_kenmerken,
    }


def parse_file(path: Path, composite_to_pair: dict) -> dict:
    """Read a cached XML file from disk and parse it."""
    xml_text = path.read_text(encoding="utf-8")
    record = parse_xml(xml_text, composite_to_pair)
    if not record["ecli"]:
        record["ecli"] = path.stem.replace("_", ":")
    return record
