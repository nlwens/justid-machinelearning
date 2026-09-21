"""Step 3 -- draw the sample from the frame. No network access.

Run with:  python download/step3_draw_sample.py

What this produces
------------------
    data/interim/sample.csv   the rulings to download, one row per ECLI

This step is deliberately separate from both the enumeration before it and
the download after it. It is pure, deterministic bookkeeping: given the same
frame and the same seed it always selects the same rulings, and it runs in
seconds. That means the sampling strategy can be changed and inspected
without re-querying the API or re-downloading anything.

Two sets are drawn, and the order matters
-----------------------------------------
1. A *natural* evaluation set, sampled uniformly at random from the entire
   frame with no balancing. It preserves the real composition of the corpus
   (roughly 47% administrative / 32% civil / 21% criminal law).

2. A *balanced* modelling set, with a fixed quota per (year, law area).

The natural set is drawn first and then excluded from the frame, so the two
can never overlap. Drawing it second would be a subtle form of leakage: the
balanced draw would have already claimed the rarest rulings, and whatever
remained for evaluation would be quietly depleted of exactly the cases the
model finds hardest.

Why two sets at all: a balanced corpus is the right shape for *training*,
because it stops the model collapsing onto the dominant area. It is the wrong
shape for *reporting*, because performance on a uniform mix is not
performance on real traffic. Keeping both means neither number has to be
defended as something it is not.
"""

from __future__ import annotations

import pandas as pd

import _paths  # noqa: F401
import config


def main() -> None:
    config.ensure_dirs()

    frame_path = config.INTERIM_DIR / "ecli_index.csv"
    if not frame_path.exists():
        raise SystemExit(f"Frame not found at {frame_path}. Run step2 first.")

    frame = pd.read_csv(frame_path)
    print(f"Frame: {len(frame):,} records, {frame['ecli'].nunique():,} rulings\n")

    # --- 1. The natural-distribution evaluation set ----------------------
    # Sampled over distinct rulings rather than over frame records, so that
    # a ruling tagged with two law areas is not twice as likely to be drawn.
    distinct_rulings = frame.drop_duplicates(subset="ecli")

    natural = distinct_rulings.sample(
        n=min(config.NATURAL_TEST_SIZE, len(distinct_rulings)),
        random_state=config.RANDOM_SEED,
    ).copy()
    natural["pool"] = "natural_test"

    print(f"Natural evaluation set: {len(natural):,} rulings")
    for area, count in natural["area"].value_counts().items():
        print(f"  {area:<30} {count:>7,}  {count / len(natural):6.1%}")

    # --- 2. The balanced modelling set -----------------------------------
    # Held-out rulings are removed from the frame before the quota draw, so
    # the two sets are disjoint by construction rather than by later checking.
    remaining = frame[~frame["ecli"].isin(set(natural["ecli"]))]

    balanced_parts: list[pd.DataFrame] = []
    shortfalls: list[str] = []

    for (year, area), group in remaining.groupby(["year", "area"], sort=True):
        quota = min(config.DOCS_PER_STRATUM, len(group))
        if quota < config.DOCS_PER_STRATUM:
            shortfalls.append(f"{year} {area}: {len(group)} available")
        if quota == 0:
            continue
        balanced_parts.append(
            group.sample(n=quota, random_state=config.RANDOM_SEED)
        )

    balanced = pd.concat(balanced_parts, ignore_index=True)

    # A ruling carrying two law areas can be drawn once per stratum. Keeping
    # it twice would over-weight multi-area rulings in training, so collapse
    # to one row per ruling.
    before_dedup = len(balanced)
    balanced = balanced.drop_duplicates(subset="ecli").copy()
    balanced["pool"] = "balanced"

    print(f"\nBalanced modelling set: {len(balanced):,} rulings")
    print(f"  ({before_dedup - len(balanced):,} duplicates removed: "
          f"rulings tagged with more than one law area)")
    for area, count in balanced["area"].value_counts().items():
        print(f"  {area:<30} {count:>7,}  {count / len(balanced):6.1%}")

    if shortfalls:
        print(f"\n{len(shortfalls)} strata below the quota of "
              f"{config.DOCS_PER_STRATUM}:")
        for line in shortfalls:
            print(f"  {line}")

    # --- 3. Write the combined download list -----------------------------
    sample = pd.concat([balanced, natural], ignore_index=True)
    sample_path = config.INTERIM_DIR / "sample.csv"
    sample.to_csv(sample_path, index=False, encoding="utf-8")

    hours = len(sample) / config.REQUESTS_PER_SECOND / 3600
    print(f"\nTotal to download: {len(sample):,} rulings")
    print(f"Estimated time at {config.REQUESTS_PER_SECOND:g} req/s: {hours:.1f} hours")
    print(f"Written to {sample_path}")


if __name__ == "__main__":
    main()
