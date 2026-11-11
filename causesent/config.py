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

Lexicon
-------
The Hedonometer list is labMT-en-v2 (hedonometer.org, 2020-03-28), the list Glass (2023) used: 10,187 words, the same
scores as labMT 1.0 (Dodds et al. 2011) for every shared word, 17 pandemic words added and 52 Twitter hashtags and
handles removed. To use the file of the first results (labMT2english.txt, 10,222 words), set
``CAUSESENT_LABMT=labMT2english.txt``.

Rule switches
-------------
Five refinements of the effect-finding rules were agreed in October 2026 (the 'v6' rules and the 30 October
decisions). They are ON by default. Each can be switched off with an environment variable set to ``0``, ``false``
or ``no``; ``CAUSESENT_RULES=main`` switches all five off and reproduces the first results' rules::

    CAUSESENT_DOUBLE_OBJECT     'caused me concerns': score the second noun (concerns), not the first (me), for every
                                verb ('made the kitchen a mess' -> mess); make / get N + N lines are left out of the
                                headline results by scoring.prep (see scoring.py)
    CAUSESENT_BARE_V_DOUBLE     'causing people pain' parsed as N + bare V: a misparsed double object, so only the
                                second word is scored (pain), except after verbs that take a bare infinitive (make,
                                let, help; and perception verbs among the baseline verbs: 'watched the kids play')
    CAUSESENT_QUANTITY          'a lot of questions': score the noun after 'of' (question), not the quantity word
    CAUSESENT_NO_PRONOUNS       do not score pronouns (it, me, this, someone, ...) in the noun slot
    CAUSESENT_CAUSE_FILTER      colloquial 'cause = because ('I left cause I was tired') is moved out of the results
    CAUSESENT_NEG_RULES         negation rules of 1 Nov 2026: a negated studied word is a negative form, kept in the
                                results ('did not result from', 'X was not due to Y', 'is not associated with' were left
                                out before), and a negative subject ('ensure that nobody gets hurt') makes a negated
                                effect. The polarity column (positive / negative form) is recorded either way
    CAUSESENT_PARSE_FIXES       two parser repairs (1 Nov 2026): (1) a participle after be / seem / appear / become /
                                remain / look ('is pretty much guaranteed to happen', 'seems guaranteed to fail') or
                                before its noun ('experimentally induced hepatitis') is a passive, like 'is guaranteed to
                                win' and 'the damage caused by' (spaCy parses these as adjectives)
    CAUSESENT_REATTACH          OFF by default. When no effect is found but a noun follows the verb directly
                                ('alcohol causes liver damage, ...' with 'damage' attached to a later verb), take that noun
                                as the object (frame 'V + N (reattached)'). COCA test 1 Nov 2026: 73 lines, about 70%
                                right (errors: '-ing' adjectives, 'a demanding job', 'an inhibiting influence')
    CAUSESENT_FORM_FIRST        look up each word's own form first and fall back to its lemma only when the form is not
                                in the lexicon ('problems' 2.92, not 'problem' 2.98; 'species', not the lemma 'specie');
                                a word with neither is skipped (decided 1 Nov 2026; off = lemma only, as in the first
                                results and in Glass's direct-object scores)
                                of 'cause' (group 'because'); see because_filter.py

Two tags are always recorded (they change no score): ``removal`` (the effect is something going away: 'make the pain
go away') and the subject of the causal verb (``subj_words``, ``subj_pos``; used by the make / get follow-up).
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
LABMT = LEXICONS / os.environ.get("CAUSESENT_LABMT", "labMT-en-v2.tsv")   # Hedonometer (labMT-en-v2; columns word, happs)
WARRINER = LEXICONS / "warriner_2013.csv"        # Warriner, Kuperman & Brysbaert (2013) valence norms

SPACY_MODEL = os.environ.get("CAUSESENT_SPACY_MODEL", "en_core_web_sm")   # results were produced with version 3.6.0

MIN_N = 30          # a mean is reported only with at least this many scored lines

_ALL_OFF = os.environ.get("CAUSESENT_RULES", "").strip().lower() == "main"


def _on(name, default=True):
    """A rule switch: on unless set to 0 / false / no (or CAUSESENT_RULES=main and the switch is not set)."""
    v = os.environ.get(name, "").strip().lower()
    if v in ("1", "true", "yes"): return True
    if v in ("0", "false", "no"): return False
    return default and not _ALL_OFF


DOUBLE_OBJECT = _on("CAUSESENT_DOUBLE_OBJECT")
BARE_V_DOUBLE = _on("CAUSESENT_BARE_V_DOUBLE")
QUANTITY = _on("CAUSESENT_QUANTITY")
NO_PRONOUNS = _on("CAUSESENT_NO_PRONOUNS")
CAUSE_FILTER = _on("CAUSESENT_CAUSE_FILTER")
FORM_FIRST = _on("CAUSESENT_FORM_FIRST")
NEG_RULES = _on("CAUSESENT_NEG_RULES")
PARSE_FIXES = _on("CAUSESENT_PARSE_FIXES")
CAUSE_FILTER_VN = _on("CAUSESENT_CAUSE_FILTER_VN", default=False)   # OFF (Dean 1 Nov, 23:42): ~4 real verbs lost per 'because' caught
SENSE_FILTERS = _on("CAUSESENT_SENSE_FILTERS")          # non-causal render / secure / save / contribute to (Dean 1 Nov)
REATTACH = _on("CAUSESENT_REATTACH", default=False)     # off: tested 1 Nov 2026, about 70% right (see below)

SWITCHES = (("double_object", DOUBLE_OBJECT), ("bare_v_double", BARE_V_DOUBLE), ("quantity", QUANTITY),
            ("no_pronouns", NO_PRONOUNS), ("cause_filter", CAUSE_FILTER), ("form_first", FORM_FIRST),
            ("neg_rules", NEG_RULES), ("parse_fixes", PARSE_FIXES), ("cause_filter_vn", CAUSE_FILTER_VN),
            ("sense_filters", SENSE_FILTERS))


def rules_label():
    """Short description of the active rules and lexicon, written into result files."""
    on = [n for n, v in SWITCHES if v]
    return ("main rules" + (" + " + ", ".join(on) if on else "")) + " | lexicon " + LABMT.name


def ensure_dirs():
    for p in (WORK, RESULTS, FIGURES):
        p.mkdir(parents=True, exist_ok=True)
