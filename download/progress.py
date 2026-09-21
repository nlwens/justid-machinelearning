"""Report download progress. Read-only -- safe to run while step 4 is going.

Run with:  .venv\\Scripts\\python.exe download\\progress.py

Measures the rate by counting cached documents twice a few seconds apart,
rather than by reading the crawler's log. That way it reports what actually
landed on disk, and it works just as well after the crawler has been
restarted, detached, or lost its terminal.
"""

from __future__ import annotations

import time

import pandas as pd

import _paths  # noqa: F401
import config

SAMPLE_INTERVAL_SECONDS = 10


def count_documents() -> int:
    """Count cached ruling XML files."""
    documents_root = config.RAW_DIR / "documents"
    if not documents_root.exists():
        return 0
    return sum(1 for _ in documents_root.rglob("*.xml"))


def main() -> None:
    sample_path = config.INTERIM_DIR / "sample.csv"
    target = len(pd.read_csv(sample_path)) if sample_path.exists() else 0

    before = count_documents()
    time.sleep(SAMPLE_INTERVAL_SECONDS)
    after = count_documents()

    rate = (after - before) / SAMPLE_INTERVAL_SECONDS

    share = f"{after / target:.1%}" if target else "n/a"
    print(f"downloaded : {after:,} / {target:,}  ({share})")
    print(f"rate       : {rate:.1f} docs/sec")

    if rate > 0:
        remaining_hours = (target - after) / rate / 3600
        print(f"status     : running, ~{remaining_hours:.1f} h remaining")
    elif target and after >= target:
        print("status     : complete")
    else:
        print("status     : STOPPED -- re-run step4_fetch_documents.py to resume")

    documents_root = config.RAW_DIR / "documents"
    if documents_root.exists():
        total_bytes = sum(p.stat().st_size for p in documents_root.rglob("*.xml"))
        print(f"on disk    : {total_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
