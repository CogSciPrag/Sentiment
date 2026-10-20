"""Scoring variants, the rules for which lines enter the results, and loading the scored lines.

The pipeline scripts store, for every line, the lemmas of the effect (N_words, V_words, ...). Scores are computed from
those lemmas here, so the four scoring variants need no re-parsing:

  raw    Hedonometer (labMT 2.0), single words, no exclusion (the main version)
  band   Hedonometer with words scoring 4-6 dropped word by word (the 'neutral band', as in Dodds et al. and Glass)
  adj    Hedonometer, with the noun's adjective and noun-compound modifiers averaged into N
         ('poor performance' -> mean(poor, performance); 'bone pain' -> mean(bone, pain))
  war    Warriner, Kuperman & Brysbaert (2013) valence norms instead of the Hedonometer

prep() keeps the lines that enter the results: the main group (not passive or negated), with an effect found;
for 'make' and 'get' only their causative frames; for blame/credit the noun after with/for is the effect, for all
other verbs the direct object.
"""
import glob
import numpy as np, pandas as pd
from . import config
from .targets import ITEMS, SYMMETRIC

# Causative frames of make and get: 'made him happy', 'made them leave', 'got the car fixed', 'got him to sign'.
# Their other uses ('make bread', 'get a job') are left out of the results (but kept in the by-frame tables).
AGG_FRAMES = {"make": ["V + N + ADJ", "V + N + bare V", "V + N + V-ed"], "get": ["V + N + ADJ", "V + N + V-ed", "V + N + to V"]}

_lex = pd.read_csv(config.LABMT, sep="\t")
H = dict(zip(_lex.word, _lex.happs))
HB = {w: h for w, h in H.items() if h > 6 or h < 4}          # neutral band (4-6) removed
_w = pd.read_csv(config.WARRINER)
WAR = dict(zip(_w.Word.str.lower(), _w["V.Mean.Sum"]))
LEX = {"raw": H, "band": HB, "adj": H, "war": WAR}
SCORINGS = ["raw", "band", "adj", "war"]
LABEL = {"raw": "raw scores", "band": "neutral band (4–6) removed", "adj": "adjectives and compounds averaged into the noun",
         "war": "Warriner et al. (2013) valence"}
BASELINE = "BASELINE (other verbs)"
CORPORA = ["ententen21", "bnc", "coca", "reddit"]


def _mean_score(words, lexicon, cache):
    """Mean lexicon score of the space-separated lemmas in `words` (unrated words skipped; NaN if none rated)."""
    if not isinstance(words, str) or not words.strip(): return np.nan
    if words in cache: return cache[words]
    v = [lexicon[w] for w in words.split() if w in lexicon]
    cache[words] = float(np.mean(v)) if v else np.nan
    return cache[words]


def _join(a, b):
    a = a.fillna("").astype(str) if a is not None else ""
    b = b.fillna("").astype(str) if b is not None else ""
    return (a + " " + b).str.strip()


def rescore(D, scoring):
    """Recompute score_N, score_V, score_NV, agg_score (and score_obj) for one scoring variant."""
    D = D.copy(); lexicon = LEX[scoring]; cache = {}
    n_words = _join(D.N_words, D["N_mods"] if "N_mods" in D else None) if scoring == "adj" else D.N_words
    D["score_N"] = n_words.map(lambda w: _mean_score(w, lexicon, cache))
    v_words = D.V_words
    if scoring == "adj" and "V_mods" in D:     # correlation items: Y is a noun, so it gets its modifiers too
        sym = D.item.isin(SYMMETRIC) if "item" in D else False
        v_words = v_words.where(~sym, _join(D.V_words, D.V_mods))
    D["score_V"] = v_words.map(lambda w: _mean_score(w, lexicon, cache))
    D["score_NV"] = D[["score_N", "score_V"]].mean(axis=1)
    has_complement = D.V_words.fillna("").astype(str).str.strip() != ""
    D["agg_score"] = np.where(has_complement, D.score_NV, D.score_N)       # the line score
    if "obj_words" in D:
        o_words = _join(D.obj_words, D["obj_mods"] if "obj_mods" in D else None) if scoring == "adj" else D.obj_words
        D["score_obj"] = o_words.map(lambda w: _mean_score(w, lexicon, cache))
    return D


def prep(D, baseline=False):
    """The lines that enter the results (see the module docstring)."""
    D = D[(D.group == "main") & ~D.frame.isin(["not matched", "no effect", "target verb"])].copy()
    if baseline:
        D["item"] = BASELINE; return D
    D = D[D.item.isin(ITEMS)]
    if "score_obj" in D:
        # 'blamed the driver for the crash' -> crash; for other verbs the direct object stays the effect
        wf = (D.frame == "V + N + with/for N") & ~D.item.isin(["blame", "credit"])
        for c in ["score_N", "score_NV", "agg_score"]: D.loc[wf, c] = D.loc[wf, "score_obj"]
        D.loc[wf, "frame"] = "V + N"
    for it, frames in AGG_FRAMES.items():
        D = D[(D.item != it) | D.frame.isin(frames)]
    return D


# ---- loading the scored lines written by the pipeline
COLS = ["corpus", "item", "frame", "group", "N_words", "V_words", "N_mods", "V_mods", "obj_words", "obj_mods", "verb",
        "prevention", "effect_left"]

def _read(f):
    return pd.read_csv(f, sep="\t", low_memory=False, usecols=lambda c: c in COLS)

def _concat(files):
    files = sorted(files)
    return pd.concat([_read(f) for f in files], ignore_index=True) if files else None

def _sketchengine_corpus(col):
    return np.where(col.astype(str).str.contains("bnc"), "bnc", "ententen21")

def load_records():
    """All scored lines: (T, B) = (studied expressions, baseline verbs), with corpus in ententen21 / bnc / coca / reddit.
    Corpora whose files are missing are skipped, so the analysis also runs on a subset."""
    W = config.WORK; T, B = [], []
    se = _concat(glob.glob(str(W / "sketchengine_scored" / "*.records.tsv")))
    if se is not None: se["corpus"] = _sketchengine_corpus(se.corpus); T.append(se)
    co = _concat(glob.glob(str(W / "coca_scored" / "*.records.tsv")))
    if co is not None: co["corpus"] = "coca"; T.append(co)
    rd = _concat(glob.glob(str(W / "reddit_scored" / "parts" / "*.tsv.gz")))
    if rd is not None: rd["corpus"] = "reddit"; T.append(rd)
    if (W / "sketchengine_baseline.records.tsv").exists():
        bs = _read(W / "sketchengine_baseline.records.tsv"); bs["corpus"] = _sketchengine_corpus(bs.corpus); B.append(bs)
    if (W / "coca_baseline.records.tsv").exists():
        bc = _read(W / "coca_baseline.records.tsv"); bc["corpus"] = "coca"; B.append(bc)
    br = _concat(glob.glob(str(W / "reddit_baseline" / "parts" / "*.tsv.gz")))
    if br is not None: br["corpus"] = "reddit"; B.append(br)
    if not T or not B:
        raise SystemExit("No scored lines found under " + str(W) + ": run the scoring steps first (see README).")
    return pd.concat(T, ignore_index=True), pd.concat(B, ignore_index=True)
