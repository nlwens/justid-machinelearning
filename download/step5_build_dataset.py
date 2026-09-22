"""Step 5 -- read the downloaded XML files into one table.

Run with:  python download/step5_build_dataset.py

What this produces
------------------
    data/interim/parsed.jsonl    every sampled ruling, one JSON object per line

This step only extracts. It does not drop short texts, does not throw away
rare labels, and does not cut train / val / test. Those decisions belong in
a notebook later, where you can see each one.

Each line looks like
--------------------
    {
      "ecli": "ECLI:NL:RVS:2007:BC1581",
      "court": "Raad van State",
      "date": "2007-12-21",
      "text": "...",
      "text_length": 6352,
      "text_source": "uitspraak",
      "rechtsgebieden": ["Bestuursrecht", "Vreemdelingenrecht"],
      "rechtsgebieden_raw": ["Bestuursrecht; Vreemdelingenrecht"],
      "bijzondere_kenmerken": ["Hoger beroep"],
      "pool": "balanced",
      "year": 2007,
      "area": "Bestuursrecht"
    }

``pool`` comes from sample.csv (balanced vs natural_test). It is a note,
not a finished split.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import _paths  # noqa: F401
import config
import parse_document


def log(message: str) -> None:
    """Print without crashing on Windows consoles that cannot show ë / ü."""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("ascii", "replace").decode("ascii"))


def load_taxonomy() -> dict:
    path = config.TAXONOMY_DIR / "rechtsgebieden.json"
    if not path.exists():
        raise SystemExit(f"Taxonomy not found at {path}. Run download/step1 first.")
    return json.loads(path.read_text(encoding="utf-8"))


def ecli_from_path(path: Path) -> str:
    return path.stem.replace("_", ":")


def main() -> None:
    config.ensure_dirs()

    sample_path = config.INTERIM_DIR / "sample.csv"
    if not sample_path.exists():
        raise SystemExit(f"Sample not found at {sample_path}. Run download/step3 first.")

    sample = pd.read_csv(sample_path)
    sample_by_ecli = sample.drop_duplicates(subset="ecli").set_index("ecli")
    wanted = set(sample_by_ecli.index)
    log(f"Sample: {len(wanted):,} rulings")

    taxonomy = load_taxonomy()
    composite_to_pair = taxonomy["composite_to_pair"]

    parsed_path = config.INTERIM_DIR / "parsed.jsonl"
    documents_root = config.RAW_DIR / "documents"
    parse_failures: list[dict] = []
    written = 0

    # Already-parsed ECLIs are skipped, so an interrupted run can continue.
    already: set[str] = set()
    if parsed_path.exists():
        with parsed_path.open(encoding="utf-8") as handle:
            for line in handle:
                already.add(json.loads(line)["ecli"])
        log(f"Already in {parsed_path.name}: {len(already):,}")

    paths = list(documents_root.rglob("*.xml")) if documents_root.exists() else []
    log(f"XML files on disk: {len(paths):,}")

    with parsed_path.open("a", encoding="utf-8") as handle:
        for path in tqdm(paths, unit="doc", desc="Parsing"):
            ecli = ecli_from_path(path)
            if ecli not in wanted or ecli in already:
                continue
            try:
                record = parse_document.parse_file(path, composite_to_pair)
            except Exception as exc:  # noqa: BLE001 - one bad file must not abort
                parse_failures.append({"ecli": ecli, "error": f"{type(exc).__name__}: {exc}"})
                continue

            sample_row = sample_by_ecli.loc[ecli]
            record["ecli"] = ecli
            record["pool"] = str(sample_row["pool"])
            record["year"] = int(sample_row["year"])
            record["area"] = str(sample_row["area"])
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            already.add(ecli)
            written += 1

    missing = sorted(wanted - already)
    log(f"\nNewly parsed: {written:,}")
    log(f"Table now has: {len(already):,}")
    log(f"Missing XML: {len(missing):,}")
    log(f"Parse failures: {len(parse_failures):,}")
    if parse_failures:
        failure_path = config.INTERIM_DIR / "parse_failures.json"
        failure_path.write_text(
            json.dumps(parse_failures, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log(f"Failures listed in {failure_path}")
    log(f"\nComplete table: {parsed_path}")


if __name__ == "__main__":
    main()
