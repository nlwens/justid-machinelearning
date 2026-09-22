"""Central configuration for the 2026 rebuild of the JustID classifier.

Every constant that more than one script needs -- where files live, which API
endpoints to call, how fast we are allowed to call them, and how the corpus is
sampled -- is declared here exactly once.

Why this file exists
--------------------
In the 2024 version of this project these settings were scattered across two
training scripts, an evaluation script and three notebooks. Each copy drifted
from the others, and the sampling logic in particular lived only inside a
notebook cell. When the source database was later deleted, the dataset became
impossible to reconstruct. Keeping the definition in one importable module is
what makes the pipeline reproducible.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
#
# Everything is resolved relative to this file, so the project can be moved or
# cloned anywhere without edits. Nothing is ever written outside BASE_DIR.
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

# Untouched API responses, exactly as received. Never edited in place: if a
# parsing bug is found later, we re-parse these instead of re-downloading.
RAW_DIR = DATA_DIR / "raw"

# Intermediate artefacts: the ECLI index, the parsed document table.
INTERIM_DIR = DATA_DIR / "interim"

# Later notebook outputs (train/val/test). Not written by the download steps.
PROCESSED_DIR = DATA_DIR / "processed"

# The official controlled vocabularies (law areas, procedure types).
TAXONOMY_DIR = DATA_DIR / "taxonomy"

ALL_DIRS = [DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR, TAXONOMY_DIR]


def ensure_dirs() -> None:
    """Create every project directory if it does not exist yet.

    Called at the start of each pipeline step so that scripts can be run in
    any order without a separate setup step.
    """
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Rechtspraak Open Data API
#
# Public, free, CC-0 licensed. Documented in "Technische documentatie Open
# Data van de Rechtspraak" (IVO v1.15).
#
# The service is split in two: you first query the ECLI *index* to find out
# which rulings match your criteria, then fetch each ruling's *content*
# separately by ECLI.
# ---------------------------------------------------------------------------

# Index endpoint. Returns an Atom feed of matching ECLIs.
SEARCH_URL = "https://data.rechtspraak.nl/uitspraken/zoeken"

# Content endpoint. Returns one ruling as XML: RDF metadata plus full text.
CONTENT_URL = "https://data.rechtspraak.nl/uitspraken/content"

# Controlled vocabularies, published as XML.
TAXONOMY_URLS = {
    # Law areas (labels for the 'rechtsgebieden' task). Nested two levels deep.
    "rechtsgebieden": "https://data.rechtspraak.nl/Waardelijst/Rechtsgebieden",
    # Procedure types (labels for the 'bijzondere kenmerken' task). Flat.
    "proceduresoorten": "https://data.rechtspraak.nl/Waardelijst/Proceduresoorten",
}

# The index endpoint refuses to return more than 1000 entries per call;
# larger result sets have to be walked with the 'from' offset parameter.
MAX_RESULTS_PER_PAGE = 1000

# Rechtspraak documents a hard ceiling of 10 requests/second and explicitly
# asks callers to leave headroom for other users. We stay below it on purpose:
# the full crawl is an overnight job either way, and getting rate-limited
# halfway through costs more time than the throttle saves.
REQUESTS_PER_SECOND = 6.0

# Number of worker threads used by the download step.
#
# This does *not* raise the request rate -- the limiter above still caps it.
# It exists because a single-threaded crawler cannot reach the configured
# rate at all: each request costs 200-400 ms of network round trip, which is
# longer than the 167 ms spacing 6 req/s asks for, so throughput collapses to
# whatever latency allows (~3 req/s) and the crawl takes twice as long.
#
# With a few workers in flight, waiting on the network overlaps and the rate
# limiter becomes the binding constraint again, which is the intended
# behaviour. More workers than this buys nothing, since the limiter is
# already saturated.
DOWNLOAD_WORKERS = 6

# Network timeouts and retry behaviour for transient failures (429, 5xx,
# dropped connections). Backoff is exponential: 2s, 4s, 8s, ...
REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 5
RETRY_BACKOFF_SECONDS = 2.0

# Sent on every request so Rechtspraak can identify the traffic. Using a
# descriptive agent rather than a spoofed browser string is both more honest
# and less likely to be blocked.
USER_AGENT = "justid-research-crawler/2026 (student project; contact via repo)"


# ---------------------------------------------------------------------------
# Corpus sampling strategy
#
# This section is the single most important difference from the 2024 pipeline.
#
# The old crawler paged through live search results with an increasing offset
# (?from=0, 1000, 2000 ...) and the dataset was then built from the *first
# 60 000 rows* of the resulting table. Two consequences followed:
#
#   1. Those first rows were the oldest rulings in the corpus, so the sample
#      was wildly unrepresentative -- 43% of it was social security law, while
#      all of criminal law accounted for 0.15%.
#   2. Offsets into a live, growing result set are not stable over time, so
#      the exact sample could never be reproduced.
#
# Instead we sample explicitly: a fixed quota per (year, top-level law area)
# stratum, drawn from date-bounded queries. Date ranges are stable -- the set
# of rulings published in 2019 does not change -- so the same configuration
# yields the same corpus on any future run.
# ---------------------------------------------------------------------------

# Years to draw from, inclusive. Full-text coverage before the mid-2000s is
# sparse and inconsistent, and the current year is still being filled in, so
# both ends are excluded deliberately.
YEAR_FROM = 2006
YEAR_TO = 2025

# Top-level law areas used as strata, identified by their stable PSI URIs
# (taken from the official Rechtsgebieden vocabulary). Sub-areas are inherited
# from the parent query, so filtering on the root is enough to stratify.
STRATUM_SUBJECTS = {
    "Bestuursrecht": "http://psi.rechtspraak.nl/rechtsgebied#bestuursrecht",
    "Civiel recht": "http://psi.rechtspraak.nl/rechtsgebied#civielRecht",
    "Strafrecht": "http://psi.rechtspraak.nl/rechtsgebied#strafrecht",
    "Internationaal publiekrecht": (
        "http://psi.rechtspraak.nl/rechtsgebied#internationaalPubliekrecht"
    ),
}

# Target number of rulings per (year, law area) cell.
#
# The frame built in step 2 contains 823 094 rulings, so the quota is a
# budget decision rather than an availability one. At the configured request
# rate, downloading costs roughly one hour per 21 600 documents:
#
#     500/stratum  ->  ~35 000 docs  ->  ~1.6 hours
#    1000/stratum  ->  ~65 000 docs  ->  ~3.0 hours   (default)
#    2000/stratum  ->  ~125 000 docs ->  ~5.8 hours
#
# 1000 is chosen because ~65 000 documents is also about the largest corpus
# that fine-tunes in a single evening on a 4 GB GPU. Raising it only helps if
# the training budget rises with it.
#
# Small strata will not reach the quota -- Internationaal publiekrecht has
# just 535 rulings in twenty years, and did not exist as a tagged area before
# 2013. Shortfalls are reported rather than silently accepted: the point is
# to know the real composition of the corpus, not to force it.
DOCS_PER_STRATUM = 1000

# Only consider ECLIs that actually carry a full-text document. Roughly half
# of all ECLIs are metadata-only registrations: for 2024 the index reports
# 154 833 ECLIs but only 73 661 with a document attached. Sampling without
# this filter wastes more than half of every request budget on empty records.
REQUIRE_FULL_DOCUMENT = True

# Size of the second, separately drawn evaluation set.
#
# The stratified sample above is deliberately *balanced* across years and law
# areas, which is the right shape for training: it stops the model from
# collapsing onto whichever area happens to dominate. But it is the wrong
# shape for reporting, because a balanced test set answers "how good is the
# model on an artificial uniform mix" rather than "how good is it on the
# rulings it will actually see".
#
# So a second set is drawn uniformly at random from the whole frame, with no
# balancing, and held out entirely. Final numbers are reported on both: the
# balanced set shows per-class capability, the natural one shows operational
# performance. Reporting only one of the two is how a model ends up looking
# much better (or much worse) on paper than it behaves in practice.
NATURAL_TEST_SIZE = 5000

# Seed for every random choice in the pipeline (stratum subsampling, dataset
# splitting). Fixed so runs are comparable.
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

# Rulings shorter than this (in characters) are almost always procedural stubs
# with no classifiable content -- withdrawals, referrals, clerical corrections.
MIN_TEXT_LENGTH = 500

# A label must appear on at least this many *training* documents to be kept.
# The 2024 pipeline used 20, which let through classes with ~25 examples out
# of 48 000; those were never predicted once and simply dragged macro-F1 down
# without contributing anything.
MIN_LABEL_FREQUENCY = 100

# Split proportions. The split is stored as lists of ECLIs so that it can be
# committed to version control and reproduced exactly.
TRAIN_FRACTION = 0.8
VAL_FRACTION = 0.1
TEST_FRACTION = 0.1


# ---------------------------------------------------------------------------
# Inference
#
# Fine-tuned weights live under models/ locally. The Docker image later loads
# the same folders, or the Hugging Face Hub ids that replace them.
# LABEL_THRESHOLD is the default cut-off used in training, eval and predict.
# ---------------------------------------------------------------------------

MODELS_DIR = BASE_DIR / "models"
RG_MODEL_DIR = MODELS_DIR / "rg"
PR_MODEL_DIR = MODELS_DIR / "pr"
RG_HUB_ID = "newnus/justid-rechtsgebieden"
PR_HUB_ID = "newnus/justid-procedures"
MAX_LEN = 512
LABEL_THRESHOLD = 0.5
