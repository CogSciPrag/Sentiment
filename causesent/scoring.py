"""Scoring variants, the rules for which lines enter the results, and loading the scored lines.

The pipeline scripts store, for every line, the lemmas of the effect (N_words, V_words, ...) and the same words as they
occur (N_forms, V_forms, ...). Scores are computed from these here, so the four scoring variants need no re-parsing.
Form-first rule (config.FORM_FIRST, decided 1 Nov 2026): each word is looked up by its own form if the lexicon lists
it, else by its lemma; a word with neither is skipped, and a line with no rated word gets no score. For 'band' the
choice of form or lemma is made in the full Hedonometer, then neutral words are dropped. Lines scored before the
forms were recorded fall back to lemmas. Variants:

  raw    Hedonometer (labMT-en-v2, Glass's list; see config.py), single words, no exclusion (the main version)
  band   Hedonometer with words scoring 4-6 dropped word by word (the 'neutral band', as in Dodds et al. and Glass)
  adj    Hedonometer, with the noun's adjective and noun-compound modifiers averaged into N
         ('poor performance' -> mean(poor, performance); 'bone pain' -> mean(bone, pain))
  war    Warriner, Kuperman & Brysbaert (2013) valence norms instead of the Hedonometer

prep() keeps the lines that enter the results: the main group (not passive, negated effect, or colloquial 'cause =
because). Each line keeps its polarity (positive / negative form of the studied word; frames.py): the results are
given for positive forms, negative forms and both combined (decided 1 Nov 2026); sample sizes refer to the combination.
with an effect found; for 'make' and 'get' only their causative frames; for blame/credit the noun after with/for is the
effect, for all other verbs the direct object.

make / get with two objects ('made him a sandwich', 'made him president', 'got me a job'; decided 30 Oct 2026):
  version B (headline, the default): no make or get N + N line counts
  version C (follow-up, prep(..., make_get_nn="C"); pipeline/11_make_get_followup.py): make N + N counts unless the
            second noun is a created object (CREATED_OBJECTS: 'made him a sandwich'); get N + N counts only when the
            subject of 'get' is clearly not a person ('having a degree won't get you a job', 'an extra skill that gets
            me raises'; not 'a friend got me a job', not imperatives, not unknown or proper-name subjects)
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
from . import config
from .targets import ITEMS, SYMMETRIC

# Causative frames of make and get: 'made him happy', 'made them leave', 'got the car fixed', 'got him to sign'.
# Their other uses ('make bread', 'get a job') are left out of the results (but kept in the by-frame tables).
AGG_FRAMES = {"make": ["V + N + ADJ", "V + N + bare V", "V + N + V-ed"], "get": ["V + N + ADJ", "V + N + V-ed", "V + N + to V"]}

# ---- make / get N + N, version C (follow-up). The three parses of a true double object; the bare-V parse
# ('I don't get why people ...') is not used here. Word lists built from 160,000 AskReddit make / get lines
# (30 Oct 2026, to be checked by Dean).
NN_FRAMES = ["V + N + N (double object)", "V + N + N (double object, clause parse)",
             "V + N + N (double object, two-object parse)"]
# things made FOR someone ('made him a sandwich', 'made me a card'): left out of make's N + N lines in version C
CREATED_OBJECTS = {"sandwich", "meal", "dinner", "food", "breakfast", "lunch", "supper", "snack", "coffee", "tea",
    "cup", "mug", "bowl", "plate", "drink", "cocktail", "pizza", "egg", "cake", "pie", "cookie", "bread", "toast", "soup",
    "sauce", "salad", "burger", "pancake", "smoothie", "costume", "card", "gift", "present", "playlist", "mixtape",
    "copy", "list", "bed", "offer", "account", "video"}
# subjects of 'get' that count as persons (besides personal pronouns and imperatives)
PERSON_NOUNS = {"person", "people", "man", "woman", "guy", "girl", "boy", "dude", "lady", "kid", "child", "baby",
    "human", "parent", "mom", "mum", "mother", "dad", "father", "grandparent", "grandmother", "grandfather", "grandma",
    "grandpa", "sister", "brother", "sibling", "aunt", "uncle", "cousin", "son", "daughter", "wife", "husband",
    "boyfriend", "girlfriend", "partner", "fiance", "fiancee", "friend", "buddy", "roommate", "neighbor", "neighbour",
    "family", "doctor", "nurse", "teacher", "professor", "student", "boss", "manager", "coworker", "colleague",
    "employer", "employee", "customer", "client", "waiter", "waitress", "player", "coach", "cop", "officer", "lawyer",
    "stranger", "sil", "troll", "mate", "bro", "gf", "bf", "ex"}
PERSON_PRONOUNS = {"i", "me", "you", "u", "ya", "y'", "he", "him", "she", "her", "we", "us", "they", "them", "one",
    "someone", "somebody", "anyone", "anybody", "everyone", "everybody", "nobody", "no one", "who", "whoever",
    "(imperative)"}
NONPERSON_PRONOUNS = {"it", "this", "that", "these", "those", "which", "what", "something", "anything",
    "everything", "nothing", "whatever"}


def subject_not_person(D):
    """True where the causal verb's subject is clearly not a person: a non-person noun or pronoun, a clause
    ('having a degree'), or an amount ('$5 gets you ...'). Persons, imperatives, proper names and unknown subjects
    give False."""
    if "subj_words" not in D:
        raise SystemExit("The scored lines have no subject columns: re-run the scoring steps with this version.")
    w = D.subj_words.fillna("").astype(str).str.lower(); p = D.subj_pos.fillna("").astype(str)
    noun = (p == "NOUN") & ~w.isin(PERSON_NOUNS)
    return noun | ((p == "PRON") & w.isin(NONPERSON_PRONOUNS)) | p.isin(["VERB", "AUX", "NUM"])


def nn_followup_keep(D):
    """Version C: the make / get N + N lines that count."""
    nn = D.frame.isin(NN_FRAMES)
    created = D.N_words.fillna("").astype(str).map(lambda s: any(x in CREATED_OBJECTS for x in s.split()))
    return nn & (((D.item == "make") & ~created) | ((D.item == "get") & subject_not_person(D)))

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
BASELINE_SCONJ = "BASELINE (other conjunctions)"     # Glass's baseline for the connectives (decided 1 Nov 2026)
SCONJ_ITEM = "other conjunctions"
BASELINE_NOUN = "BASELINE (other nouns)"             # baseline for the noun items (decided 1 Nov 2026)
NOUN_ITEM = "other nouns"
CORPORA = ["ententen21", "bnc", "coca", "reddit"]


MEMBER = {"raw": H, "band": H, "adj": H, "war": WAR}     # where the form-first rule looks for the form


def keys(words, forms, member):
    """The strings looked up for the space-separated lemmas `words` and their forms `forms` (form-first rule)."""
    l = words.split() if isinstance(words, str) else []
    if not config.FORM_FIRST or not isinstance(forms, str): return l
    f = forms.split()
    if len(f) != len(l): return l          # not aligned (should not happen): lemmas only
    return [a if a in member else b for a, b in zip(f, l)]


def _mean_score(words, forms, lexicon, member, cache):
    """Mean lexicon score of the words (form-first rule; unrated words skipped; NaN if none rated)."""
    if not isinstance(words, str) or not words.strip(): return np.nan
    ck = (words, forms if isinstance(forms, str) else None)
    if ck in cache: return cache[ck]
    v = [lexicon[w] for w in keys(words, forms, member) if w in lexicon]
    cache[ck] = float(np.mean(v)) if v else np.nan
    return cache[ck]


def scored_keys(D, words_col, forms_col):
    """Space-separated looked-up strings per line (Hedonometer membership), e.g. for the word-shift figures."""
    f = D[forms_col] if forms_col in D else pd.Series(None, index=D.index, dtype=object)
    return pd.Series([" ".join(keys(w, x, H)) for w, x in zip(D[words_col], f)], index=D.index)


def _join(a, b):
    a = a.fillna("").astype(str) if a is not None else ""
    b = b.fillna("").astype(str) if b is not None else ""
    return (a + " " + b).str.strip()


def _col(D, c):
    return D[c] if c in D else pd.Series(None, index=D.index, dtype=object)


def _pairs(D, words, forms, mods=None, mods_forms=None):
    """(lemmas, forms) per line, with the modifiers appended when given; forms are dropped (None) for a line whose
    forms are missing, so that line is scored by its lemmas."""
    w, f = _col(D, words), _col(D, forms)
    if mods is not None:
        m, mf = _col(D, mods), _col(D, mods_forms)
        ok = f.notna() & (mf.notna() | m.fillna("").astype(str).str.strip().eq(""))
        w, f = _join(w, m), _join(f, mf).where(ok, None)
    return list(zip(w, f))


def rescore(D, scoring):
    """Recompute score_N, score_V, score_NV, agg_score (and score_obj) for one scoring variant."""
    D = D.copy(); lexicon = LEX[scoring]; member = MEMBER[scoring]; cache = {}
    sc = lambda pairs: [_mean_score(w, f, lexicon, member, cache) for w, f in pairs]
    adj = scoring == "adj"
    D["score_N"] = sc(_pairs(D, "N_words", "N_forms", *(("N_mods", "N_mods_forms") if adj else ())))
    vp = _pairs(D, "V_words", "V_forms")
    if adj and "V_mods" in D:     # correlation items: Y is a noun, so it gets its modifiers too
        sym = D.item.isin(SYMMETRIC).values if "item" in D else np.zeros(len(D), bool)
        vm = _pairs(D, "V_words", "V_forms", "V_mods", "V_mods_forms")
        vp = [b if s else a for a, b, s in zip(vp, vm, sym)]
    D["score_V"] = sc(vp)
    D["score_NV"] = D[["score_N", "score_V"]].mean(axis=1)
    has_complement = D.V_words.fillna("").astype(str).str.strip() != ""
    D["agg_score"] = np.where(has_complement, D.score_NV, D.score_N)       # the line score
    if "obj_words" in D:
        D["score_obj"] = sc(_pairs(D, "obj_words", "obj_forms", *(("obj_mods", "obj_mods_forms") if adj else ())))
    return D


def prep(D, baseline=False, make_get_nn="B"):
    """The lines that enter the results (see the module docstring). make_get_nn: "B" (headline) or "C" (follow-up)."""
    D = D[(D.group == "main") & ~D.frame.isin(["not matched", "no effect", "target verb"])].copy()
    if baseline:
        D["item"] = np.select([D.item.eq(SCONJ_ITEM), D.item.eq(NOUN_ITEM)], [BASELINE_SCONJ, BASELINE_NOUN], BASELINE)
        return D
    D = D[D.item.isin(ITEMS)]
    if "score_obj" in D:
        # 'blamed the driver for the crash' -> crash; for other verbs the direct object stays the effect
        wf = (D.frame == "V + N + with/for N") & ~D.item.isin(["blame", "credit"])
        for c in ["score_N", "score_NV", "agg_score"]: D.loc[wf, c] = D.loc[wf, "score_obj"]
        D.loc[wf, "frame"] = "V + N"
    extra = nn_followup_keep(D) if make_get_nn == "C" else pd.Series(False, index=D.index)
    for it, frames in AGG_FRAMES.items():
        D = D[(D.item != it) | D.frame.isin(frames) | extra.loc[D.index]]
    return D


# ---- loading the scored lines written by the pipeline
COLS = ["corpus", "item", "frame", "group", "N_words", "V_words", "N_mods", "V_mods", "obj_words", "obj_mods", "verb",
        "prevention", "effect_left", "removal", "subj_words", "subj_pos", "cause_reading", "polarity",
        "N_forms", "V_forms", "N_mods_forms", "V_mods_forms", "obj_forms", "obj_mods_forms", "conj",
        "sense", "avoid_V", "noun"]
POLARITIES = ["combined", "positive", "negative"]
def by_polarity(D):
    """(label, lines) for the combination and the two forms. Lines scored before polarity was recorded count only in
    'combined'."""
    pol = D["polarity"] if "polarity" in D else pd.Series("", index=D.index)
    return [("combined", D), ("positive", D[pol == "positive"]), ("negative", D[pol == "negative"])]

def _read(f):
    D = pd.read_csv(f, sep="\t", low_memory=False, usecols=lambda c: c in COLS)
    D["sample"] = Path(f).name.split(".")[0]          # e.g. se_coca__get__form_got
    return D


# ---- COCA make / get: per-form samples (decided Oct / 1 Nov 2026). COCA's 500-line lemma samples have too few causative
# make / get lines (get: 18). Extra samples of about 500 lines per word form were downloaded (coca/raw/
# coca_batch_make_get_forms_20261101.txt). They are combined in proportion to each form's frequency in COCA: a fixed-seed
# subsample of each form sample, as large as the smallest share allows (all lines of the most frequent form are used), so
# the combined sample has COCA's mix of forms and needs no weights. The lemma samples are added as they are.
FORM_FREQ = {"get": {"get": 1743864, "gets": 157876, "got": 1060061, "gotten": 57888, "getting": 327522},
             "make": {"make": 1023498, "makes": 262882, "made": 706701, "making": 297473}}
FORM_SEED = 20261101

def proportional_forms(D):
    """Subsample the per-form samples (sample name '<...>__<item>__form_<form>') in proportion to FORM_FREQ."""
    if "sample" not in D: return D
    keep = pd.Series(True, index=D.index)
    for item, freq in FORM_FREQ.items():
        tot = sum(freq.values()); share = {f: n / tot for f, n in freq.items()}
        lines = {f: D.index[D["sample"].astype(str).str.endswith(f"__{item}__form_{f}")] for f in freq}
        have = {f: len(ix) for f, ix in lines.items() if len(ix)}
        if not have: continue
        total = min(have[f] / share[f] for f in have)          # largest total the samples allow
        rng = np.random.default_rng(FORM_SEED)
        for f, ix in lines.items():
            if not len(ix): continue
            k = min(len(ix), int(round(total * share[f])))
            chosen = set(rng.choice(np.asarray(ix), size=k, replace=False).tolist())
            keep.loc[ix] = [i in chosen for i in ix]
    return D[keep]

# ---- COCA items searched with several queries (2 Nov 2026): COCA's search syntax has no optional slot, so 'in (the) light
# of', 'as a (ADJ) result', 'the (ADJ) result of', 'in order to / that / for, so as to' ... were searched separately (one
# sample per query, sample names '<...>__<item>__q<i>'; queries and hits in docs/queries.tsv). The samples are combined in
# proportion to the queries' hits in COCA, like the make / get forms. Lines that belong to another expression (group
# 'overlap': 'as a result of' in a 'the|a|an result of' sample) are dropped first, and each query's hits are scaled by the
# share of its lines that are not overlap, so the combination has COCA's mix of the item's own lines.
QUERY_FREQ = {  # COCA hits per query (2 Nov 2026), keyed by the sample-name suffix 'q<i>'; item -> {suffix: hits}
    "as a consequence (of)": {"q1": 3515, "q2": 208},
    "in light of": {"q1": 9313, "q2": 2601},
    "in order to": {"q1": 91548, "q2": 1124, "q3": 5121, "q4": 6537},
    "result of": {"q1": 60577, "q2": 3934},
    "consequence of": {"q1": 10724, "q2": 4320},
    "as a result (of)": {"q1": 47363, "q2": 550},
}
QUERY_SEED = 20261102

def _slug(item):
    import re
    return re.sub(r"[^A-Za-z]+", "_", item)

def proportional_queries(D):
    """Combine the COCA samples of an item searched with several queries, in proportion to QUERY_FREQ (see above)."""
    if "sample" not in D: return D
    keep = pd.Series(True, index=D.index)
    sname = D["sample"].astype(str); grp = D["group"].astype(str) if "group" in D else pd.Series("", index=D.index)
    for item, freq in QUERY_FREQ.items():
        lines = {q: D.index[sname.str.endswith(f"__{_slug(item)}__{q}")] for q in freq}
        if not any(len(ix) for ix in lines.values()): continue
        own = {q: ix[grp.loc[ix].ne("overlap").values] for q, ix in lines.items()}
        for q, ix in lines.items(): keep.loc[ix] = False                 # re-selected below
        eff = {q: freq[q] * (len(own[q]) / len(lines[q])) for q in freq if len(lines[q])}
        tot = sum(eff.values())
        if not tot: continue
        share = {q: e / tot for q, e in eff.items()}
        total = min(len(own[q]) / share[q] for q in share if share[q] > 0)
        rng = np.random.default_rng(QUERY_SEED)
        for q in share:
            ix = own[q]; k = min(len(ix), int(round(total * share[q])))
            if k: keep.loc[rng.choice(np.asarray(ix), size=k, replace=False)] = True
    return D[keep]

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
    if co is not None:
        co["corpus"] = "coca"; co = proportional_queries(proportional_forms(co.reset_index(drop=True))); T.append(co)
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
