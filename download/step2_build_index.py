"""Step 2 -- build the sampling frame: every candidate ruling, before sampling.

Run with:  python download/step2_build_index.py

What this produces
------------------
    data/interim/index/<year>_<area>.json   one file per stratum (resumable)
    data/interim/ecli_index.csv             the combined frame

The frame is a table of (ecli, year, area). It contains *no document text* --
only identifiers -- so it is small and fast to rebuild, and it can be
committed to version control as a record of exactly what the sample was drawn
from.

Why the frame is built exhaustively
-----------------------------------
It is tempting to ask the API for just the 750 rulings each stratum needs and
be done in a fraction of the requests. That would reintroduce the exact bug
this rebuild exists to fix.

The index returns results in a fixed order, so "the first 750 rulings of 2019
in criminal law" is not a sample of 2019 -- it is a sample of early January
2019. The 2024 pipeline made precisely this mistake at a larger scale by
taking the first 60 000 rows of its database, and ended up with a corpus that
was 43% social security law and 0.15% criminal law.

So we enumerate every candidate first and draw the sample from the complete
frame in step 3. Enumerating costs one request per 1000 ECLIs -- a few
minutes in total -- and in exchange the sample is both unbiased within each
stratum and reproducible, because the frame is stored rather than re-queried.
"""

from __future__ import annotations

import json

import pandas as pd

import _paths  # noqa: F401
import config
import rechtspraak_api


def stratum_path(year: int, area: str) -> "object":
    """Return the cache file for one (year, area) stratum.

    Each stratum is written as soon as it is fetched, so an interrupted run
    resumes where it stopped instead of starting over.
    """
    safe_area = area.replace(" ", "_")
    return config.INTERIM_DIR / "index" / f"{year}_{safe_area}.json"


def fetch_stratum(year: int, area: str, subject_uri: str) -> list[str]:
    """Return every ECLI in one (year, area) stratum, using the disk cache."""
    path = stratum_path(year, area)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["eclis"]

    # `date` filters on the ruling date (uitspraakdatum), and the pair of
    # values is read by the API as an inclusive range.
    eclis = rechtspraak_api.search_eclis(
        date_from=f"{year}-01-01",
        date_to=f"{year}-12-31",
        subject_uri=subject_uri,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"year": year, "area": area, "subject_uri": subject_uri, "eclis": eclis},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return eclis


def main() -> None:
    config.ensure_dirs()

    years = list(range(config.YEAR_FROM, config.YEAR_TO + 1))
    areas = config.STRATUM_SUBJECTS

    print(
        f"Enumerating {len(years)} years x {len(areas)} law areas "
        f"= {len(years) * len(areas)} strata\n"
    )

    rows: list[dict] = []
    # Reported per year so a run in progress shows whether coverage looks
    # sane, rather than only revealing problems at the very end.
    for year in years:
        counts: list[str] = []
        for area, subject_uri in areas.items():
            eclis = fetch_stratum(year, area, subject_uri)
            for ecli in eclis:
                rows.append({"ecli": ecli, "year": year, "area": area})
            counts.append(f"{area[:12]}={len(eclis)}")
        print(f"  {year}  " + "  ".join(counts))

    frame = pd.DataFrame(rows)

    # A ruling tagged with several law areas legitimately appears in several
    # strata. Keep those rows: the area column is what step 3 stratifies on,
    # and document-level de-duplication happens once the sample is drawn.
    output_path = config.INTERIM_DIR / "ecli_index.csv"
    frame.to_csv(output_path, index=False, encoding="utf-8")

    unique_rulings = frame["ecli"].nunique()
    print(
        f"\nFrame: {len(frame):,} (year, area) records covering "
        f"{unique_rulings:,} distinct rulings"
    )
    print(f"Written to {output_path}")

    print("\nComposition by law area (share of all records):")
    for area, count in frame["area"].value_counts().items():
        print(f"  {area:<30} {count:>8,}  {count / len(frame):6.1%}")

    # The headline comparison for the writeup: the 2024 corpus was 43% social
    # security law and 0.15% criminal law purely because of how it was drawn.
    print(
        "\nFor reference, the 2024 corpus was 43% social security law "
        "and 0.15% criminal law."
    )


if __name__ == "__main__":
    main()
