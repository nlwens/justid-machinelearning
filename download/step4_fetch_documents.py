"""Step 4 -- download the sampled rulings. This is the long one.

Run with:  python download/step4_fetch_documents.py

What this produces
------------------
    data/raw/documents/<COURT>/ECLI_....xml   one file per ruling
    data/interim/fetch_failures.json          ECLIs that could not be fetched

Expect roughly three hours for the default 65 480-ruling sample, and about
1.5 GB on disk.

Safe to interrupt
-----------------
Every document is written to disk as soon as it arrives, and the script skips
anything already cached. Stopping it with Ctrl-C and restarting tomorrow
costs one directory scan, not another three hours. That is a deliberate
design choice: a crawl you are afraid to interrupt is a crawl you only run
once, and a pipeline you only run once is one you cannot fix.

Nothing here parses or interprets the XML. That happens in
step5_build_dataset.py, reading from this cache.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

import _paths  # noqa: F401
import config
import rechtspraak_api


def fetch_one(ecli: str) -> tuple[str, str | None]:
    """Fetch a single ruling, converting any failure into a return value.

    Failures are returned rather than raised so that one unreachable document
    out of sixty-five thousand does not abort the entire run. The failed
    ECLIs are written out at the end and can be retried by simply running the
    script again -- successful ones are cached and will be skipped.

    :returns: ``(ecli, None)`` on success, ``(ecli, error_message)`` on failure.
    """
    try:
        rechtspraak_api.fetch_document(ecli)
        return ecli, None
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
        return ecli, f"{type(exc).__name__}: {exc}"


def main() -> None:
    config.ensure_dirs()

    sample_path = config.INTERIM_DIR / "sample.csv"
    if not sample_path.exists():
        raise SystemExit(f"Sample not found at {sample_path}. Run step3 first.")

    sample = pd.read_csv(sample_path)
    eclis = sample["ecli"].tolist()

    # Count what is already cached so the progress bar reflects real work
    # rather than restarting from zero on every resume.
    cached = sum(1 for _, _ in rechtspraak_api.iter_cached_documents())
    print(f"Sample: {len(eclis):,} rulings")
    print(f"Already cached: {cached:,}")
    print(f"Workers: {config.DOWNLOAD_WORKERS} at {config.REQUESTS_PER_SECOND:g} req/s\n")

    failures: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=config.DOWNLOAD_WORKERS) as executor:
        futures = {executor.submit(fetch_one, ecli): ecli for ecli in eclis}

        with tqdm(total=len(futures), unit="doc", smoothing=0.05) as progress:
            for future in as_completed(futures):
                ecli, error = future.result()
                if error is not None:
                    failures[ecli] = error
                progress.update(1)
                progress.set_postfix(failed=len(failures))

    failures_path = config.INTERIM_DIR / "fetch_failures.json"
    failures_path.write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    documents_root = config.RAW_DIR / "documents"
    total_bytes = sum(p.stat().st_size for p in documents_root.rglob("*.xml"))

    print(f"\nDownloaded. {len(failures):,} failures "
          f"({len(failures) / len(eclis):.2%})")
    print(f"Cache now holds {total_bytes / 1e9:.2f} GB in {documents_root}")
    if failures:
        print(f"Failed ECLIs listed in {failures_path}")
        print("Re-run this script to retry them; cached documents are skipped.")


if __name__ == "__main__":
    main()
