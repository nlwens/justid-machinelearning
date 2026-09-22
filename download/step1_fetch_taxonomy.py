"""Step 1 -- download the official Rechtspraak label vocabularies.

Run with:  python download/step1_fetch_taxonomy.py

What this produces
------------------
    data/taxonomy/rechtsgebieden.xml     raw response, kept for provenance
    data/taxonomy/proceduresoorten.xml   idem
    data/taxonomy/rechtsgebieden.json    parsed hierarchy
    data/taxonomy/proceduresoorten.json  parsed flat list

Why this step exists
--------------------
The 2024 pipeline never fetched these lists. It derived its label set purely
from whatever values happened to occur in its (biased) sample, which caused
two problems.

First, the hierarchy was lost. Rechtspraak tags a ruling with a value like
"Bestuursrecht; Socialezekerheidsrecht", meaning the sub-area
*Socialezekerheidsrecht* within the parent area *Bestuursrecht*. The old code
treated that whole string as one atomic class, unrelated to the class
"Bestuursrecht". The model therefore could not learn that predicting the
child implies the parent, and the two competed as if they were alternatives.

Second, the label set was an accident of sampling. Areas absent from the
sample simply did not exist as far as the model was concerned, and there was
no way to tell "this class is genuinely rare" apart from "we never drew one".

The official vocabulary fixes both: it gives the real, complete tree, with
stable URIs that also serve as the stratification keys in step 2.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import _paths  # noqa: F401
import config
import rechtspraak_api


def parse_rechtsgebieden(xml_text: str) -> dict:
    """Parse the law-area vocabulary into a flat, queryable structure.

    The source XML nests sub-areas inside their parent::

        <Rechtsgebied>
          <Identifier>...#bestuursrecht</Identifier>
          <Naam>Bestuursrecht</Naam>
          <Rechtsgebied>
            <Identifier>...#bestuursrecht_ambtenarenrecht</Identifier>
            <Naam>Ambtenarenrecht</Naam>
          </Rechtsgebied>
          ...

    Note that a child's ``<Naam>`` is the bare sub-area name
    ("Ambtenarenrecht"), whereas rulings are tagged with the *composite*
    string "Bestuursrecht; Ambtenarenrecht". Both forms are recorded, because
    the composite is what we have to match against document metadata while
    the parent/child pair is what we actually want to model.

    :returns: a dict with three keys -- ``roots`` (top-level area names),
        ``areas`` (one entry per node), and ``composite_to_pair`` (the lookup
        used when parsing documents in step 4).
    """
    root = ET.fromstring(xml_text)

    roots: list[str] = []
    areas: list[dict] = []
    composite_to_pair: dict[str, list[str | None]] = {}

    # The vocabulary XML declares only xsi/xsd namespaces, so element tags are
    # unprefixed and can be matched directly by name.
    for parent_element in root.findall("Rechtsgebied"):
        parent_name = (parent_element.findtext("Naam") or "").strip()
        parent_uri = (parent_element.findtext("Identifier") or "").strip()
        if not parent_name:
            continue

        roots.append(parent_name)
        areas.append(
            {
                "name": parent_name,
                "uri": parent_uri,
                "parent": None,
                "composite": parent_name,
                "level": 1,
            }
        )
        # A ruling can be tagged with the parent area alone.
        composite_to_pair[parent_name] = [parent_name, None]

        for child_element in parent_element.findall("Rechtsgebied"):
            child_name = (child_element.findtext("Naam") or "").strip()
            child_uri = (child_element.findtext("Identifier") or "").strip()
            if not child_name:
                continue

            # This is the exact form that appears in a document's
            # <dcterms:subject> field.
            composite = f"{parent_name}; {child_name}"

            areas.append(
                {
                    "name": child_name,
                    "uri": child_uri,
                    "parent": parent_name,
                    "composite": composite,
                    "level": 2,
                }
            )
            composite_to_pair[composite] = [parent_name, child_name]

    return {
        "roots": roots,
        "areas": areas,
        "composite_to_pair": composite_to_pair,
    }


def parse_proceduresoorten(xml_text: str) -> dict:
    """Parse the procedure-type vocabulary (the 'bijzondere kenmerken' labels).

    This list is flat, but the element naming is not guaranteed to match the
    law-area file, so every descendant carrying a ``<Naam>`` is collected
    rather than assuming a fixed tag name.
    """
    root = ET.fromstring(xml_text)

    items: list[dict] = []
    seen: set[str] = set()

    for element in root.iter():
        name = (element.findtext("Naam") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        items.append(
            {
                "name": name,
                "uri": (element.findtext("Identifier") or "").strip(),
            }
        )

    return {"bijzondere_kenmerken": items}


def main() -> None:
    config.ensure_dirs()

    # --- Law areas -------------------------------------------------------
    print("Downloading Rechtsgebieden vocabulary ...")
    rechtsgebieden_xml = rechtspraak_api.fetch_taxonomy("rechtsgebieden")
    (config.TAXONOMY_DIR / "rechtsgebieden.xml").write_text(
        rechtsgebieden_xml, encoding="utf-8"
    )

    rechtsgebieden = parse_rechtsgebieden(rechtsgebieden_xml)
    (config.TAXONOMY_DIR / "rechtsgebieden.json").write_text(
        json.dumps(rechtsgebieden, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    top_level = [a for a in rechtsgebieden["areas"] if a["level"] == 1]
    sub_level = [a for a in rechtsgebieden["areas"] if a["level"] == 2]
    print(
        f"  {len(top_level)} top-level areas, {len(sub_level)} sub-areas "
        f"({len(rechtsgebieden['composite_to_pair'])} distinct labels)"
    )
    for area in top_level:
        children = [a["name"] for a in sub_level if a["parent"] == area["name"]]
        print(f"    {area['name']} ({len(children)} sub-areas)")

    # --- Procedure types -------------------------------------------------
    print("\nDownloading Proceduresoorten vocabulary ...")
    proceduresoorten_xml = rechtspraak_api.fetch_taxonomy("proceduresoorten")
    (config.TAXONOMY_DIR / "proceduresoorten.xml").write_text(
        proceduresoorten_xml, encoding="utf-8"
    )

    proceduresoorten = parse_proceduresoorten(proceduresoorten_xml)
    (config.TAXONOMY_DIR / "proceduresoorten.json").write_text(
        json.dumps(proceduresoorten, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  {len(proceduresoorten['bijzondere_kenmerken'])} bijzondere kenmerken")

    # --- Cross-check the stratification keys in config -------------------
    # config.STRATUM_SUBJECTS hardcodes four PSI URIs. If Rechtspraak ever
    # renames one, step 2 would silently return zero results for that whole
    # stratum, so verify them against the vocabulary we just downloaded.
    known_uris = {area["uri"] for area in rechtsgebieden["areas"]}
    print("\nVerifying stratum URIs declared in config.py ...")
    for label, uri in config.STRATUM_SUBJECTS.items():
        status = "ok" if uri in known_uris else "NOT FOUND IN VOCABULARY"
        print(f"  {label:<30} {status}")

    print(f"\nWritten to {config.TAXONOMY_DIR}")


if __name__ == "__main__":
    main()
