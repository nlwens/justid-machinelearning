"""Smoke test for the assumptions this pipeline makes about the API.

Run with:  python download/check_api.py

Every check below corresponds to something the crawler silently depends on.
If one of them ever stops holding -- Rechtspraak changes a parameter name,
caps paging at some offset, renames a vocabulary URI -- the crawler would not
crash. It would just quietly collect less data than intended, and the problem
would surface weeks later as an inexplicably weak model.

That is not hypothetical: the 2024 pipeline's central flaw was exactly this
kind of silent truncation, and it went unnoticed until after the project
ended. Ten seconds of verification is cheap insurance.
"""

from __future__ import annotations

import _paths  # noqa: F401
import config
import rechtspraak_api


def check(description: str, passed: bool, detail: str = "") -> bool:
    """Print one check result in a fixed format and return whether it passed."""
    mark = "PASS" if passed else "FAIL"
    print(f"  [{mark}] {description}")
    if detail:
        print(f"         {detail}")
    return passed


def main() -> None:
    results: list[bool] = []

    # --- 1. Paging returns everything the index claims exists -------------
    # The crawler walks pages until it sees an empty one. If the API capped
    # the reachable offset, the walk would end early and the frame would be
    # short -- with no error raised anywhere.
    print("\n1. Paging completeness")
    date_from, date_to = "2024-03-01", "2024-03-31"
    reported = rechtspraak_api.count_eclis(date_from, date_to)
    collected = rechtspraak_api.search_eclis(date_from, date_to)
    # A small shortfall is acceptable: the feed occasionally carries entries
    # that are not rulings, which search_eclis filters out.
    shortfall = reported - len(collected)
    results.append(
        check(
            "paging reaches the full result set",
            0 <= shortfall <= reported * 0.01,
            f"index reports {reported:,}, paging collected {len(collected):,} "
            f"(shortfall {shortfall})",
        )
    )

    # --- 2. return=DOC actually filters out metadata-only records ---------
    # Roughly half of all ECLIs have no full text. Sampling them would waste
    # the request budget and produce empty training examples.
    print("\n2. Full-document filter")
    config.REQUIRE_FULL_DOCUMENT = False
    without_filter = rechtspraak_api.count_eclis(date_from, date_to)
    config.REQUIRE_FULL_DOCUMENT = True
    with_filter = rechtspraak_api.count_eclis(date_from, date_to)
    results.append(
        check(
            "return=DOC narrows the result set",
            0 < with_filter < without_filter,
            f"{without_filter:,} ECLIs total, {with_filter:,} with a document "
            f"({with_filter / without_filter:.0%})",
        )
    )

    # --- 3. Date ranges are stable and disjoint ---------------------------
    # The whole reproducibility argument rests on this: a past date range
    # always describes the same set of rulings. Two adjacent months must sum
    # to the range that spans them.
    print("\n3. Date ranges partition cleanly")
    january = rechtspraak_api.count_eclis("2024-01-01", "2024-01-31")
    february = rechtspraak_api.count_eclis("2024-02-01", "2024-02-29")
    both = rechtspraak_api.count_eclis("2024-01-01", "2024-02-29")
    results.append(
        check(
            "adjacent months sum to their span",
            january + february == both,
            f"{january:,} + {february:,} = {january + february:,} vs {both:,}",
        )
    )

    # --- 4. Law-area filtering works and is inclusive of sub-areas --------
    # Stratification assumes that filtering on a top-level area also returns
    # rulings tagged with its sub-areas. If it did not, three quarters of
    # every stratum would be missing.
    print("\n4. Law-area stratification keys")
    unfiltered = rechtspraak_api.count_eclis(date_from, date_to)
    per_area_total = 0
    for area, uri in config.STRATUM_SUBJECTS.items():
        count = rechtspraak_api.count_eclis(date_from, date_to, subject_uri=uri)
        per_area_total += count
        results.append(
            check(f"{area} returns results", count > 0, f"{count:,} rulings")
        )
    # The per-area counts should roughly account for the unfiltered total.
    # They can exceed it, because a ruling may carry several law areas.
    results.append(
        check(
            "law areas together cover the corpus",
            per_area_total >= unfiltered * 0.9,
            f"areas sum to {per_area_total:,} vs {unfiltered:,} unfiltered "
            f"(overlap is expected: rulings can have several areas)",
        )
    )

    # --- 5. A fetched document carries both label fields and a body -------
    # If the metadata fields were missing or renamed, every document would
    # parse to an empty label set, and the dataset would look valid while
    # being entirely unlabelled.
    print("\n5. Document structure")
    sample_ecli = collected[0]
    xml_text = rechtspraak_api.fetch_document(sample_ecli, use_cache=False)
    results.append(
        check(
            "document carries a law area (dcterms:subject)",
            "dcterms:subject" in xml_text,
            f"checked {sample_ecli}",
        )
    )
    results.append(
        check(
            "document carries a procedure type (psi:procedure)",
            "psi:procedure" in xml_text,
        )
    )
    results.append(
        check(
            "document carries full text (<uitspraak>)",
            "<uitspraak" in xml_text,
            f"{len(xml_text):,} characters of XML",
        )
    )

    # --- Summary ---------------------------------------------------------
    passed = sum(results)
    print(f"\n{passed}/{len(results)} checks passed")
    if passed < len(results):
        print("Do not start the crawl until the failures above are understood.")


if __name__ == "__main__":
    main()
