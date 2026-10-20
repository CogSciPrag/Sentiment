"""Finding the effect of a causal expression in a parsed sentence, tagging its frame, and scoring it.

Main entry points
-----------------
find_all(doc, only_items=None)   every studied expression in a spaCy Doc, with its effect
analyse(doc, item, triggers)     one expression, given the token(s) where it occurs

Each result is a dict (one "record") with:
  item           the studied expression ('cause', 'because of', ...)
  frame          the construction it occurs in, e.g. 'V + N', 'V + N + to V', 'V + that-clause',
                 'effect on the left (clause)', 'passive'; 'no effect' when no effect could be identified
  group          'main' (enters the results), 'passive' or 'negated' (reported separately), or 'active'/'other'
                 (correlation items only)
  N_words        lemma(s) of the effect's head noun (or the subject of an effect clause); coordinated nouns are listed
  V_words        lemma(s) of the effect's verb, adjective or predicate, when the effect has one
  N_mods         adjectives and noun compounds modifying N (used by the 'adj' scoring variant)
  score_N, score_V, score_NV, agg_score
                 Hedonometer scores: N, V, their mean, and the line score (N+V mean if there is a complement,
                 else N). These are the raw scores; scoring.py recomputes them for the other variants.
  prevention     True for prevention words (the effect is what does NOT happen)
  effect_left    True when the effect precedes the expression (result from, because of, ...)

The rules themselves, with examples, are documented in docs/scoring_rules.md.
"""
import numpy as np, pandas as pd
from . import config
from .targets import ITEMS, PREVENTION, EFFECT_LEFT

_lex = pd.read_csv(config.LABMT, sep="\t")
H = dict(zip(_lex.word, _lex.happs))          # lemma -> Hedonometer happiness score (1-9)
NEG = {"not", "never", "n't", "no", "nt"}
SUBJ = ("nsubj", "nsubjpass", "csubj")

def lem(t): return t.lemma_.lower()
def conj_group(t):
    """t plus its coordinated siblings of the same kind ('fires and floods', 'pack and leave')."""
    out = [t]
    for c in t.children:
        if c.dep_ == "conj" and c.pos_ in (t.pos_, "PROPN", "NOUN", "VERB", "ADJ"):
            out.extend(conj_group(c))
    return out
def score(tokens):
    vals = [H[lem(t)] for t in tokens if lem(t) in H]
    return float(np.mean(vals)) if vals else np.nan
def is_neg(t): return any(c.dep_ == "neg" or c.lower_ in NEG and c.dep_ in ("advmod", "neg", "det") for c in t.children)

MOD_DEPS = ("amod", "compound")
def mods_of(tokens):
    """Adjectival and noun-compound modifiers of the given nouns ('poor performance', 'bone pain'), as lemmas."""
    out = []
    for t in tokens:
        for c in t.children:
            if c.dep_ in MOD_DEPS and c.pos_ in ("ADJ", "VERB", "NOUN", "PROPN"):
                out.append(c)
                out.extend(x for x in c.children if x.dep_ == "conj" and x.pos_ == c.pos_)   # 'safe and happy'
    return out

# ---- Optional rule: quantity words (CAUSESENT_QUANTITY=1; off by default)
# A head noun that names an amount, a part or a kind is replaced by the noun of its of-phrase
# ('a lot of questions' -> question, 'part of its popularity' -> popularity, 'a kind of organization' -> organization).
QUANT_ON = config.QUANTITY
# Pure amounts: replaced by their of-noun, and NOT scored when there is no of-phrase ('22 percent', 'much more').
QUANT_DROP = {"lot", "lots", "plenty", "amount", "bit", "couple", "dozen", "hundred", "thousand", "million", "billion",
 "trillion", "percent", "percentage", "majority", "minority", "half", "most", "some", "more", "much", "many", "all", "any",
 "each", "both", "few", "several", "none", "one"}
# Part / kind / degree words: replaced by their of-noun, but kept as they are without one ('metal parts', 'a degree').
QUANT_OF = {"number", "deal", "total", "sum", "host", "ton", "load", "handful", "bunch", "rest", "bulk", "quarter", "third",
 "part", "portion", "share", "fraction", "kind", "sort", "type", "form", "variety", "range", "series", "string", "array",
 "level", "degree", "extent"}
def quant_through(t, depth=0):
    if t is None or not QUANT_ON: return t
    is_num = t.pos_ == "NUM" or t.like_num
    if lem(t) in QUANT_DROP or lem(t) in QUANT_OF or is_num:
        of = next((c for c in t.children if c.dep_ == "prep" and c.lower_ == "of"), None)
        po = next((c for c in of.children if c.dep_ == "pobj"), None) if of is not None else None
        if po is not None and depth <= 3: return quant_through(po, depth + 1)
        return None if (lem(t) in QUANT_DROP or is_num) else t
    return t
def _resolve(group):
    out = []
    for x in group:
        y = quant_through(x)
        if y is None: continue
        out.extend(conj_group(y) if y is not x else [y])
    return out

# ---- Optional rule: no pronouns (CAUSESENT_NO_PRONOUNS=1; off by default)
# Pronouns in the N slot are not scored, subjects and objects alike ('it stemmed from', 'ensure that you ...',
# 'caused me to ...'), including demonstratives, relatives and indefinites. The line then rests on V; with no V it
# gets no score.
NOPRON_ON = config.NO_PRONOUNS
PRONOUNS = {"i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself", "yourselves", "ya", "u", "he", "him", "his",
 "himself", "she", "her", "hers", "herself", "it", "its", "itself", "we", "us", "our", "ours", "ourselves", "they", "them",
 "their", "theirs", "themselves", "em", "'em", "one", "oneself", "this", "that", "these", "those", "which", "who", "whom",
 "whose", "what", "whoever", "whomever", "whatever", "whichever", "everyone", "everybody", "someone", "somebody",
 "anyone", "anybody", "no one", "nobody", "nothing", "everything", "something", "anything", "none"}
def _nopron(tokens):
    return [t for t in tokens if lem(t) not in PRONOUNS and t.lower_ not in PRONOUNS] if NOPRON_ON else tokens

def rec(item, frame, N=None, V=None, group="main"):
    Nt = _nopron(_resolve(conj_group(N))) if N is not None else []
    Vt = _resolve(conj_group(V)) if V is not None else []
    sN, sV = score(Nt), score(Vt)
    vals = [x for x in (sN, sV) if not np.isnan(x)]
    sNV = float(np.mean(vals)) if vals else np.nan
    has_comp = V is not None
    return dict(item=item, frame=frame, group=group, prevention=item in PREVENTION or ITEMS[item][0] in PREVENTION,
                effect_left=item in EFFECT_LEFT,
                N_words=" ".join(lem(t) for t in Nt), V_words=" ".join(lem(t) for t in Vt),
                N_mods=" ".join(lem(t) for t in mods_of(Nt)),
                V_mods=" ".join(lem(t) for t in mods_of([v for v in Vt if v.pos_ in ("NOUN", "PROPN")])),
                score_N=sN, score_V=sV, score_NV=sNV, agg_score=(sNV if has_comp else sN))

def subject_of(v):
    return next((c for c in v.children if c.dep_ in SUBJ), None)
def predicate(v):
    """Main verb, or the predicate after a copula ('the roads were icy' -> icy)."""
    if lem(v) == "be":
        p = next((c for c in v.children if c.dep_ in ("acomp", "attr")), None)
        if p is not None: return p
    return v

def clause_effect(item, v, frame):
    """Effect is a clause headed by v: N = its subject, V = its verb/predicate."""
    grp = "negated" if is_neg(v) else "main"
    return rec(item, frame, subject_of(v), predicate(v), grp)

DOUBLE_OBJ_EXCLUDE = {"make", "get", "render", "let"}
def _adverbial_np(n):
    """'a lot', 'the most', 'a bit', 'the whole show': degree / duration phrases, not a second object."""
    if lem(n) in ("guy", "all", "both", "each"):   # 'you guys', 'us all': apposition, not a second object
        return True
    if lem(n) in ("lot", "bit", "most", "more", "less", "much", "little", "while", "deal") and not any(g.dep_ == "prep" and g.lower_ == "of" for g in n.children):
        return True
    return any(g.lower_ == "whole" for g in n.children) and lem(n) in ("show", "time", "day", "night", "week", "year", "game", "season", "way")

def complement_of(t, item=None):
    """Returns (frame, N, V, negated) for a verb-like trigger t, or None."""
    kids = list(t.children)
    dobj = next((c for c in kids if c.dep_ in ("dobj", "dative") and c.i > t.i), None)
    # ---- Optional rule: double objects (CAUSESENT_DOUBLE_OBJECT=1; off by default)
    # 'caused me concerns' -> score the second noun (the thing caused), not the indirect object. Not for make, get and
    # render ('made him president', 'render it a loser' are resultatives) or let.
    if config.DOUBLE_OBJECT and lem(t) not in DOUBLE_OBJ_EXCLUDE:
        d2 = next((c for c in kids if c.dep_ == "dobj" and c.i > t.i), None)
        if (dobj is not None and dobj.dep_ == "dative" and dobj.pos_ in ("PRON", "PROPN", "NOUN") and d2 is not None
                and d2.i > dobj.i and not _adverbial_np(d2)):
            return ("V + N + N (double object)", d2, None, False)
        # spaCy's other parse of the same construction: 'caused them problems' = ccomp NOUN 'problems' with nsubj 'them'
        if dobj is None:
            for c in kids:
                if (c.dep_ == "ccomp" and c.i > t.i and c.tag_ in ("NN", "NNS", "NNP", "NNPS") and not _adverbial_np(c)
                        and not any(g.dep_ in ("aux", "xcomp", "ccomp", "acl", "relcl") for g in c.children)):
                    s0 = next((g for g in c.children if g.dep_ in SUBJ and t.i < g.i < c.i), None)
                    # the two nouns must directly follow the verb: V [N1] [N2 ...], no comma, no 'that'
                    if (s0 is not None and s0.pos_ in ("PRON", "PROPN", "NOUN") and s0.tag_ != "VB"
                            and s0.left_edge.i == t.i + 1 and len(list(s0.subtree)) <= 4
                            and not any(y.dep_ == "prep" for y in s0.subtree)
                            and all(x.head.i == c.i or x.head.i in {y.i for y in c.subtree if y.i < c.i and y.i > s0.right_edge.i}
                                    for x in c.doc[s0.right_edge.i + 1: c.i])
                            and not any(x.is_punct for x in c.doc[t.i + 1: c.i])):
                        return ("V + N + N (double object, clause parse)", c, None, False)
    # clausal / infinitival / adjectival / participial complements
    for c in kids:
        if c.i < t.i or c.dep_ not in ("xcomp", "ccomp", "oprd", "acomp"): continue
        csubj = next((g for g in c.children if g.dep_ in SUBJ and g.i < c.i), None)
        has_to = any(g.lower_ == "to" and g.dep_ == "aux" for g in c.children)
        mark_that = any(g.dep_ == "mark" and g.lower_ in ("that",) for g in c.children)
        neg = is_neg(c)
        if c.pos_ == "ADJ" or (c.dep_ in ("oprd", "acomp") and c.pos_ != "VERB"):
            N = dobj if dobj is not None else csubj
            if N is None: continue
            return ("V + N + ADJ", N, c, neg)
        if c.tag_ == "VBN" and not any(g.dep_ in ("aux", "auxpass", "mark") for g in c.children):
            N = dobj if dobj is not None else csubj
            if N is not None: return ("V + N + V-ed", N, c, neg)
        finite = c.tag_ in ("VBZ", "VBD", "VBP", "MD") or any(g.dep_ == "aux" and g.tag_ in ("MD", "VBZ", "VBD", "VBP") for g in c.children)
        if c.tag_ == "VB" and not has_to and not mark_that and (dobj is not None or csubj is not None):
            return ("V + N + bare V", dobj if dobj is not None else csubj, c, neg)
        contiguous = c.left_edge.i > t.i and not any(x.is_punct for x in c.doc[t.i + 1: c.i])
        if c.dep_ == "ccomp" and contiguous and (mark_that or (csubj is not None and finite)) and not has_to:
            return ("V + that-clause", csubj, predicate(c), neg)
        N = dobj if dobj is not None else csubj
        if has_to:
            return (("V + N + to V" if N is not None else "V + to V"), N, c, neg)
        if c.tag_ == "VBN":
            return ("V + N + V-ed", N, c, neg) if N is not None else None
        if c.tag_ == "VB" and N is not None:
            return ("V + N + bare V", N, c, neg)
    # from / for / in + V-ing ; with / for + N
    preps = [c for c in kids if c.dep_ == "prep"] + ([c for c in dobj.children if c.dep_ == "prep" and c.lower_ in ("with", "for", "from")] if dobj is not None else [])
    for c in preps:
        if c.lower_ in ("from", "for", "in", "with"):
            g = next((x for x in c.children if x.dep_ in ("pcomp", "pobj")), None)
            if g is None: continue
            if g.tag_ == "VBG" and c.lower_ in ("from", "for", "in"):
                return (f"V + N + {c.lower_} V-ing" if dobj is not None else f"V + {c.lower_} V-ing", dobj, g, is_neg(g))
            if g.pos_ in ("NOUN", "PROPN") and c.lower_ in ("with", "for") and dobj is not None and item in ("blame", "credit"):
                return ("V + N + with/for N", g, None, False, dobj)
    if dobj is not None:
        return ("V + N", dobj, None, False)
    return None

def prep_object(p):
    """Object of preposition p: ('noun', N) | ('ving', subj, V) | None."""
    g = next((x for x in p.children if x.dep_ in ("pobj", "pcomp")), None)
    if g is None: return None
    if g.tag_ == "VBG" or g.pos_ == "VERB":
        return ("ving", subject_of(g) or next((x for x in g.children if x.dep_ in ("poss",)), None), g)
    ving = next((x for x in g.children if x.dep_ in ("acl", "relcl") and x.tag_ == "VBG" and x.i == g.i + 1), None)
    if ving is not None:
        return ("ving", g, ving)
    return ("noun", g)

def analyse(doc, item, triggers):
    """triggers: tokens in doc that are candidate triggers for item (lemma/word already matched)."""
    lemma, kind = ITEMS[item]
    out = []
    for t in triggers:
        try:
            r = _one(doc, item, lemma, kind, t)
        except Exception:
            r = None
        if r is not None:
            r["trigger_i"] = t.i
            out.append(r)
    return out

def _one(doc, item, lemma, kind, t):
    kids = list(t.children)
    if kind == "verb":
        if t.pos_ not in ("VERB", "AUX") and not t.tag_.startswith("VB"): return None
        if item == "make":   # 'make sure' and 'make a difference to' are separate items
            if any(c.lower_ == "sure" for c in kids): return None
            if any(c.dep_ == "dobj" and c.lemma_.lower() == "difference" for c in kids): return None
        if item == "require" and any(c.dep_ == "prep" and c.lower_ == "for" for c in kids) and t.tag_ == "VBN": return None
        passive = any(c.dep_ in ("nsubjpass", "auxpass", "agent") for c in kids) or (t.dep_ == "acl" and t.tag_ == "VBN")
        if passive:
            N = next((c for c in kids if c.dep_ == "nsubjpass"), None)
            if N is None and t.dep_ == "acl": N = t.head
            comp = next((c for c in kids if c.dep_ in ("xcomp", "ccomp") and c.i > t.i), None)
            return rec(item, "passive", N, comp, "passive")
        cf = complement_of(t, item)
        if cf is None:
            if t.dep_ == "relcl" and t.head.pos_ in ("NOUN", "PROPN"):   # 'the damage they cause': object relativised
                return rec(item, "V + N", t.head)
            return rec(item, "no effect")
        frame, N, V, neg = cf[:4]
        r = rec(item, frame, N, V, "negated" if neg else "main")
        if len(cf) > 4:   # with/for-N frame: also keep the direct object's score
            ot = _nopron(conj_group(cf[4])); r["obj_words"] = " ".join(lem(x) for x in ot); r["score_obj"] = score(ot)
            r["obj_mods"] = " ".join(lem(x) for x in mods_of(ot))
        return r
    if kind == "impl":
        if not t.tag_.startswith("VB"): return None
        if item in ("manage to", "fail to"):
            c = next((c for c in kids if c.dep_ in ("xcomp",) and any(g.lower_ == "to" for g in c.children)), None)
            if c is None: return None
            return rec(item, "V + to V", None, c, "negated" if is_neg(c) else "main")
        p = next((c for c in kids if c.dep_ == "prep" and c.lower_ == "in"), None)
        if p is None: return None
        po = prep_object(p)
        if po is None: return None
        if po[0] == "noun": return rec(item, "V + PREP + N", po[1])
        return rec(item, "V + PREP + V-ing", po[1], po[2], "negated" if is_neg(po[2]) else "main")
    if kind == "phrasal":
        if not (t.tag_.startswith("VB") or t.pos_ in ("VERB", "AUX")): return None
        if item == "make sure":
            sure = next((c for c in kids if c.lower_ == "sure"), None)
            if sure is None: return None
            c = next((g for g in list(sure.children) + kids if g.dep_ in ("ccomp", "xcomp") and g.i > sure.i), None)
            if c is None: return rec(item, "no effect")
            if any(g.lower_ == "to" and g.dep_ == "aux" for g in c.children):
                return rec(item, "V + to V", None, c, "negated" if is_neg(c) else "main")
            return rec(item, "V + that-clause", subject_of(c), predicate(c), "negated" if is_neg(c) else "main")
        if item == "make a difference to":
            d = next((c for c in kids if c.dep_ == "dobj" and lem(c) == "difference"), None)
            if d is None: return None
            if any(g.lower_ == "all" for g in d.children) or not any(g.lower_ == "a" for g in d.children): return None
            p = next((c for c in list(d.children) + kids if c.dep_ == "prep" and c.lower_ == "to"), None)
            if p is None: return None
            po = prep_object(p)
            if po is None: return None
            return rec(item, "V + PREP + N", po[1]) if po[0] == "noun" else rec(item, "V + PREP + V-ing", po[1], po[2])
        if item == "bring about":
            if not any(c.lower_ == "about" and c.dep_ in ("prt", "prep", "advmod") for c in kids): return None
            cf = complement_of(t)
            if cf is None:   # 'about' parsed as preposition: its object is the effect
                p = next((c for c in kids if c.lower_ == "about"), None)
                po = prep_object(p) if p is not None else None
                if po is None: return rec(item, "no effect")
                return rec(item, "V + N", po[1]) if po[0] == "noun" else rec(item, "V + PREP + V-ing", po[1], po[2])
            frame, N, V, neg = cf[:4]
            return rec(item, frame, N, V, "negated" if neg else "main")
        if item == "give rise to":
            rise = next((c for c in kids if c.dep_ == "dobj" and lem(c) == "rise"), None)
            if rise is None: return None
            p = next((c for c in list(rise.children) + kids if c.dep_ == "prep" and c.lower_ == "to"), None)
            po = prep_object(p) if p is not None else None
            if po is None: return rec(item, "no effect")
            return rec(item, "V + PREP + N", po[1]) if po[0] == "noun" else rec(item, "V + PREP + V-ing", po[1], po[2])
        prepw = {"contribute to": "to", "lead to": "to", "result in": "in", "result from": "from", "stem from": "from",
                 "arise from": "from", "depend on": ("on", "upon"), "account for": "for"}[item]
        prepw = prepw if isinstance(prepw, tuple) else (prepw,)
        p = next((c for c in kids if c.dep_ in ("prep", "prt") and c.lower_ in prepw), None)
        if p is None: return None
        if item in EFFECT_LEFT:     # effect = the subject (or the noun the participle modifies)
            if any(c.dep_ in ("nsubjpass", "auxpass") for c in kids): return rec(item, "passive", None, None, "passive")
            N = subject_of(t)
            if N is None and t.dep_ in ("acl", "relcl") and t.head.pos_ in ("NOUN", "PROPN", "PRON"): N = t.head
            if N is not None and N.dep_ == "relcl": N = N.head
            if N is not None and N.lower_ in ("which", "that", "who") and t.dep_ == "relcl": N = t.head
            return rec(item, "effect on the left", N, None, "negated" if is_neg(t) else "main")
        if any(c.dep_ in ("nsubjpass", "auxpass") for c in kids): return rec(item, "passive", subject_of(t), None, "passive")
        po = prep_object(p)
        if po is None: return rec(item, "no effect")
        if po[0] == "noun": return rec(item, "V + PREP + N", po[1])
        return rec(item, "V + PREP + V-ing", po[1], po[2], "negated" if is_neg(po[2]) else "main")
    if kind == "conn":
        if item == "because (clause)":
            if t.lower_ != "because" or t.dep_ != "mark": return None
            sub = t.head                                # head of the because-clause
            main = sub.head if sub.dep_ in ("advcl", "ccomp", "conj") and sub.head is not sub else None
            if main is None or main.pos_ not in ("VERB", "AUX", "ADJ", "NOUN"): return None
            return clause_effect(item, main, "effect on the left (clause)")
        words = {"because of": ["because"], "due to": ["due"], "thanks to": ["thanks"], "owing to": ["owing"],
                 "as a result of": ["result"]}[item]
        if t.lower_ not in words: return None
        nxt = doc[t.i + 1] if t.i + 1 < len(doc) else None
        need = {"because of": "of", "due to": "to", "thanks to": "to", "owing to": "to", "as a result of": "of"}[item]
        if nxt is None or nxt.lower_ != need: return None
        if item == "as a result of" and not (t.i >= 2 and doc[t.i - 1].lower_ == "a" and doc[t.i - 2].lower_ == "as"): return None
        anchor = doc[t.i - 2] if item == "as a result of" else t
        h = anchor.head
        steps = 0
        while h.i in range(anchor.i, nxt.i + 1) and h.head is not h and steps < 4: h = h.head; steps += 1
        if h.pos_ in ("NOUN", "PROPN", "PRON") and anchor.dep_ in ("amod", "prep", "acl"):
            return rec(item, "effect on the left (noun)", h)
        if lem(h) == "be" and predicate(h) is not h and predicate(h).i not in range(anchor.i, nxt.i + 1):
            return clause_effect(item, h, "effect on the left (clause)")
        if lem(h) == "be" and anchor.dep_ in ("acomp", "attr", "prep"):
            return rec(item, "effect on the left (noun)", subject_of(h), None, "negated" if is_neg(h) else "main")
        if h.pos_ in ("VERB", "AUX", "ADJ"):
            return clause_effect(item, h, "effect on the left (clause)")
        return None
    if kind == "assoc":
        # X ... associated/correlated with Y. X -> N slot, Y -> V slot (both scored; see SYMMETRIC in targets.py)
        vb = "associated" if item == "associated with" else "correlated"; base = vb[:-1]
        p = next((c for c in kids if c.dep_ == "prep" and c.lower_ == "with"), None)
        if p is None:   # 'correlated the scores with income': the parser often hangs 'with' on the object
            p = next((g for c in kids if c.dep_ == "dobj" for g in c.children if g.dep_ == "prep" and g.lower_ == "with"), None)
        if p is None: return rec(item, "no effect")
        po = prep_object(p)
        if po is None: return rec(item, "no effect")
        Y = po[2] if po[0] == "ving" else po[1]
        neg = is_neg(t) or (t.dep_ == "acomp" and is_neg(t.head))
        grp = "negated" if neg else "main"
        active = t.pos_ == "VERB" and t.tag_ != "VBN" and not any(c.dep_ in ("nsubjpass", "auxpass") for c in kids)
        dob = next((c for c in kids if c.dep_ == "dobj" or (active and item == "associated with" and c.dep_ == "npadvmod" and c.pos_ == "NOUN")), None)
        if dob is not None and not any(c.dep_ in ("nsubjpass", "auxpass") for c in kids):
            X = dob          # 'people associate summer with holidays'
            return rec(item, f"{base} X with Y (active)", X, Y, "active")
        if t.dep_ in ("acl", "amod") and t.head.pos_ in ("NOUN", "PROPN", "PRON") and not any(c.dep_ in ("nsubj", "nsubjpass") for c in kids):
            return rec(item, f"N {vb} with Y", t.head, Y, grp)  # 'the risks associated with smoking'
        if active and item == "associated with":      # 'I don't associate with them': social sense, not correlation
            return rec(item, "associate with N (social sense)", None, Y, "other")
        X = next((c for c in kids if c.dep_ in ("nsubjpass", "nsubj")), None)
        if X is None and t.dep_ in ("acomp", "xcomp", "conj"): X = subject_of(t.head)
        if X is None: return rec(item, "with Y only", None, Y, grp)
        frame = f"X {base}s with Y" if t.pos_ == "VERB" and not any(c.dep_ in ("auxpass",) for c in kids) and t.tag_ != "VBN" else f"X is {vb} with Y"
        return rec(item, frame, X, Y, grp)
    if kind == "adj":
        w = {"sufficient for": "sufficient", "necessary for": "necessary", "required for": "required", "enough for": "enough"}[item]
        if t.lower_ != w: return None
        cands = kids + [c for c in t.head.children if c.i > t.i and t.head is not t]
        p = next((c for c in cands if c.dep_ == "prep" and c.lower_ == "for"), None)
        vfor = next((c for c in cands if c.pos_ in ("VERB", "AUX") and any(g.lower_ == "for" and g.dep_ in ("mark", "prep") for g in c.children)), None)
        if vfor is not None:   # 'sufficient for the plant to grow'
            N = next((g for g in vfor.children if g.dep_ in SUBJ), None)
            return rec(item, "ADJ + for N to V", N, vfor, "negated" if is_neg(vfor) else "main")
        if p is None: return None
        po = prep_object(p)
        if po is None: return None
        if po[0] == "noun":
            inner = next((g for g in po[1].children if g.dep_ in ("acl", "relcl") and any(x.lower_ == "to" for x in g.children)), None)
            if inner is not None: return rec(item, "ADJ + for N to V", po[1], inner, "negated" if is_neg(inner) else "main")
            return rec(item, "ADJ + for N", po[1])
        return rec(item, "ADJ + for V-ing", po[1], po[2], "negated" if is_neg(po[2]) else "main")
    return None

# ---- trigger finding
_BY_LEMMA = {}
for _it, (_lem, _kind) in ITEMS.items():
    _BY_LEMMA.setdefault(_lem, []).append(_it)
_WORD_TRIG = {"because": ["because of", "because (clause)"], "due": ["due to"], "thanks": ["thanks to"], "owing": ["owing to"],
              "sufficient": ["sufficient for"], "necessary": ["necessary for"], "required": ["required for"], "enough": ["enough for"],
              "result": ["as a result of"],
              "associated": ["associated with"], "associate": ["associated with"], "associates": ["associated with"], "associating": ["associated with"],
              "correlated": ["correlated with"], "correlate": ["correlated with"], "correlates": ["correlated with"], "correlating": ["correlated with"]}
def find_all(doc, only_items=None):
    out = []
    for t in doc:
        cands = []
        low, lm = t.lower_, lem(t)
        if low in _WORD_TRIG: cands += _WORD_TRIG[low]
        if t.tag_.startswith("VB") or t.pos_ in ("VERB", "AUX"):
            for it in _BY_LEMMA.get(lm, []):
                if ITEMS[it][1] in ("verb", "phrasal", "impl"): cands.append(it)
        for it in dict.fromkeys(cands):
            if only_items and it not in only_items: continue
            out.extend(analyse(doc, it, [t]))
    return out
