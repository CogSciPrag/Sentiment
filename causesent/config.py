"""Paths and rule switches, in one place.

Paths
-----
Everything the pipeline reads or writes lives under one data folder, ``data/`` next to this repository by default.
Set the environment variable ``CAUSESENT_DATA`` to use another location::

    data/
      raw/              corpus samples you download yourself (not distributed; see README)
        sketchengine/   enTenTen21 and BNC concordances (Sketch Engine CSV exports, see docs/queries.tsv)
        coca/           COCA concordances (english-corpora.org, see docs/queries.tsv)
        reddit/         RC_2018-01.zst (Pushshift / Arctic Shift dump of January 2018 Reddit comments)
      work/             intermediate files written by the pipeline (parsed and scored lines)
      results/          tables written by the analysis scripts
      figures/          figures

The two lexicons go in ``lexicons/`` (also overridable with ``CAUSESENT_LEXICONS``).

Rule switches
-------------
Three refinements of the effect-finding rules were added after the main results were produced (October 2026).
They are OFF by default, so the code reproduces the main results. Switch them on with environment variables
(any of ``1``, ``true``, ``yes``)::

    CAUSESENT_DOUBLE_OBJECT=1   'caused me concerns': score the second noun (concerns), not the first (me)
    CAUSESENT_QUANTITY=1        'a lot of questions': score the noun after 'of' (question), not the quantity word
    CAUSESENT_NO_PRONOUNS=1     do not score pronouns (it, me, this, someone, ...) in the noun slot

See docs/scoring_rules.md for the details of each rule.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

DATA = Path(os.environ.get("CAUSESENT_DATA", REPO / "data"))
RAW = DATA / "raw"
WORK = DATA / "work"
RESULTS = DATA / "results"
FIGURES = DATA / "figures"

LEXICONS = Path(os.environ.get("CAUSESENT_LEXICONS", REPO / "lexicons"))
LABMT = LEXICONS / "labMT2english.txt"          # Hedonometer (labMT 2.0, English)
WARRINER = LEXICONS / "warriner_2013.csv"        # Warriner, Kuperman & Brysbaert (2013) valence norms

SPACY_MODEL = os.environ.get("CAUSESENT_SPACY_MODEL", "en_core_web_sm")   # results were produced with version 3.6.0

MIN_N = 30          # a mean is reported only with at least this many scored lines


def _on(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


DOUBLE_OBJECT = _on("CAUSESENT_DOUBLE_OBJECT")
QUANTITY = _on("CAUSESENT_QUANTITY")
NO_PRONOUNS = _on("CAUSESENT_NO_PRONOUNS")


def rules_label():
    """Short description of the active rule switches, written into result files."""
    on = [n for n, v in (("double_object", DOUBLE_OBJECT), ("quantity", QUANTITY), ("no_pronouns", NO_PRONOUNS)) if v]
    return "main rules" + (" + " + ", ".join(on) if on else "")


def ensure_dirs():
    for p in (WORK, RESULTS, FIGURES):
        p.mkdir(parents=True, exist_ok=True)
