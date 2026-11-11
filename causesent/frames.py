"""Finding the effect of a causal expression in a parsed sentence, tagging its frame, and scoring it.

Main entry points
-----------------
find_all(doc, only_items=None)   every studied expression in a spaCy Doc, with its effect
analyse(doc, item, triggers)     one expression, given the token(s) where it occurs

Each result is a dict (one "record") with:
  item           the studied expression ('cause', 'because of', ...)
  frame          the construction it occurs in, e.g. 'V + N', 'V + N + to V', 'V + that-clause',
                 'effect on the left (clause)', 'passive'; 'no effect' when no effect could be identified
  group          'main' (enters the results), 'passive' or 'negated' (reported separately), 'other sense' (a
                 non-causal sense of render / secure / save / contribute to; left out), 'because' (colloquial
                 'cause = because, left out; see because_filter.py), or 'active'/'other' (correlation items only)
  N_words        lemma(s) of the effect's head noun (or the subject of an effect clause); coordinated nouns are listed
  V_words        lemma(s) of the effect's verb, adjective or predicate, when the effect has one
  N_mods         adjectives and noun compounds modifying N (used by the 'adj' scoring variant)
  N_forms, V_forms, N_mods_forms, V_mods_forms, obj_forms, obj_mods_forms
                 the same words as they occur (lowercased), one per lemma in the same order: with the form-first rule
                 (config.FORM_FIRST) a word is scored by its form when the lexicon has it, else by its lemma
  score_N, score_V, score_NV, agg_score
                 Hedonometer scores: N, V, their mean, and the line score (N+V mean if there is a complement,
                 else N). These are the raw scores; scoring.py recomputes them for the other variants.
  prevention     True for prevention words (the effect is what does NOT happen)
  effect_left    True when the effect precedes the expression (result from, because of, ...)
  removal        True when the effect is something going away ('make the pain go away', 'helped the war end');
                 a tag only, no line is excluded (word list REMOVAL below)
  subj_words, subj_pos   verbs only: lemma and part of speech of the causal verb's subject ('A friend got me a job'
                 -> friend, NOUN; '(imperative)' for imperatives); used by the make / get follow-up in scoring.py
  cause_reading  'cause' only, bare form: the word-order rule's reading (verb / because / other; because_filter.py)
  polarity       'positive' or 'negative': whether the studied word ITSELF is negated (decided 1 Nov 2026). Negative =
                 not / n't / never on the word ('didn't cause', 'does not ensure', 'is not sufficient for'), or a
                 negative subject ('nothing causes', 'no one forced me', 'no study found', 'none of them led to'); for
                 the connectives, 'X was not due to / because of Y' ('be' + CONN). Negation inside the effect
                 ('persuaded him not to go') is not polarity: such lines stay in group 'negated' (out of the results).
                 Every line gets a polarity, so results can be given for positive forms, negative forms and both.
                 'no' / 'zero' on the effect noun ('caused no damage', 'led to no good', 'caused me no harm') also makes
                 a negative form (decided 1 Nov 2026), like 'didn't cause damage' and 'nothing caused damage'.

The rules themselves, with examples, are documented in docs/scoring_rules.md.
"""
import numpy as np, pandas as pd
from . import config
from .targets import ITEMS, PREVENTION, EFFECT_LEFT, SCONJ_EXCLUDE, SCONJ_EXCLUDE_AFTER, NOUN_BASE_EXCLUDE
from .because_filter import classify as cause_reading, is_because_line
from .senses import causal_sense

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
def key(t):
    """The string looked up in the Hedonometer: the word's own form if listed (form-first rule, 1 Nov 2026), else its lemma."""
    return t.lower_ if config.FORM_FIRST and t.lower_ in H else lem(t)
def forms(tokens): return " ".join(t.lower_.replace(" ", "_") for t in tokens)
def score(tokens):
    vals = [H[key(t)] for t in tokens if key(t) in H]
    return float(np.mean(vals)) if vals else np.nan
def is_neg(t): return any(c.dep_ == "neg" or c.lower_ in NEG and c.dep_ in ("advmod", "neg", "det") for c in t.children)

# ---- Polarity of the studied word itself (decided 1 Nov 2026): positive vs negative forms, tracked per line
NEG_SUBJ = {"nothing", "nobody", "noone", "none", "neither", "nought", "naught"}
def neg_subject(v):
    """v's subject is negative: 'nothing', 'nobody', 'none (of ...)', 'neither', 'no one', 'no study'. A verb coordinated
    without its own subject takes the first conjunct's ('no one saw or caused it')."""
    s = next((c for c in v.children if c.dep_ in SUBJ), None)
    if s is None and v.dep_ == "conj": s = next((c for c in v.head.children if c.dep_ in SUBJ), None)
    if s is None: return False
    return s.lower_ in NEG_SUBJ or any(c.dep_ == "det" and c.lower_ in ("no", "neither") for c in s.children)
def word_neg(t):
    """The studied word t is negated: not / n't / never on it, or a negative subject. For a predicate adjective or
    participle the copula counts too ('is not sufficient for', 'was not associated with')."""
    hosts = [t]
    if t.dep_ in ("acomp", "attr") and t.head is not t: hosts.append(t.head)
    return any(is_neg(h) or neg_subject(h) for h in hosts)
def polarity(neg): return "negative" if neg else "positive"
def eff_neg(v):
    """Negation INSIDE the effect (group 'negated'): not / n't / never on the effect's verb, or a negative subject
    ('ensure that nobody gets hurt', 'make sure nothing breaks'; same negative subjects as word_neg, 1 Nov 2026)."""
    return is_neg(v) or (config.NEG_RULES and neg_subject(v))

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

# ---- Rule switch: quantity words (CAUSESENT_QUANTITY; on by default)
# A head noun that names an amount, a part or a kind is replaced by the noun of its of-phrase
# ('a lot of questions' -> question, 'part of its popularity' -> popularity, 'a kind of organization' -> organization).
QUANT_ON = config.QUANTITY
# Pure amounts: replaced by their of-noun, and NOT scored when there is no of-phrase ('22 percent', 'much more').
# Money words added 1 Nov 2026 (Dean): '3 million dollars of damage' -> damage (before: dollar 7.30).
QUANT_DROP = {"lot", "lots", "plenty", "amount", "bit", "couple", "dozen", "hundred", "thousand", "million", "billion",
 "trillion", "percent", "percentage", "majority", "minority", "half", "most", "some", "more", "much", "many", "all", "any",
 "each", "both", "few", "several", "none", "one",
 "dollar", "buck", "cent", "penny", "pound", "quid", "euro", "yen", "yuan", "rupee", "peso", "franc", "grand", "$", "£", "€"}
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

# ---- Rule switch: no pronouns (CAUSESENT_NO_PRONOUNS; on by default)
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

# ---- Tag (always on, changes no score): the effect is something going away
# Word list agreed 29 Oct 2026, confirmed by Dean 1 Nov 2026: go away, disappear, stop, vanish, fade, end, pass.
REMOVAL = {"disappear", "stop", "vanish", "fade", "end", "pass"}
def _removal(N, V):
    if V is not None:
        if lem(V) in REMOVAL: return True
        return lem(V) == "go" and any(c.lower_ == "away" and c.dep_ in ("prt", "advmod") for c in V.children)
    # 'let the pain fade', 'helped the war end': spaCy often parses the bare verb as a noun with a compound
    return N is not None and lem(N) in REMOVAL and any(c.dep_ == "compound" for c in N.children)

NO_DET = {"no", "zero"}
def _no_det(n):
    """'no damage', 'zero studies': the effect noun carries 'no' / 'zero' (a negative form of the word; 1 Nov 2026)."""
    return n is not None and any(c.lower_ in NO_DET and c.dep_ in ("det", "neg", "nummod", "amod", "advmod") for c in n.children)

def rec(item, frame, N=None, V=None, group="main"):
    effect_no = _no_det(N)
    Nt = _nopron(_resolve(conj_group(N))) if N is not None else []
    Vt = _resolve(conj_group(V)) if V is not None else []
    sN, sV = score(Nt), score(Vt)
    vals = [x for x in (sN, sV) if not np.isnan(x)]
    sNV = float(np.mean(vals)) if vals else np.nan
    has_comp = V is not None
    return dict(item=item, frame=frame, group=group, effect_no=effect_no,
                prevention=item in PREVENTION or ITEMS[item][0] in PREVENTION,
                effect_left=item in EFFECT_LEFT, removal=_removal(N, V),
                N_words=" ".join(lem(t) for t in Nt), V_words=" ".join(lem(t) for t in Vt),
                N_mods=" ".join(lem(t) for t in mods_of(Nt)),
                V_mods=" ".join(lem(t) for t in mods_of([v for v in Vt if v.pos_ in ("NOUN", "PROPN")])),
                N_forms=forms(Nt), V_forms=forms(Vt), N_mods_forms=forms(mods_of(Nt)),
                V_mods_forms=forms(mods_of([v for v in Vt if v.pos_ in ("NOUN", "PROPN")])),
                score_N=sN, score_V=sV, score_NV=sNV, agg_score=(sNV if has_comp else sN))

def subject_of(v):
    return next((c for c in v.children if c.dep_ in SUBJ), None)
def predicate(v):
    """Main verb, or the predicate after a copula ('the roads were icy' -> icy). 'be going to V' -> V (2 Nov 2026)."""
    if lem(v) == "go" and v.tag_ == "VBG" and any(c.dep_ == "aux" and lem(c) == "be" for c in v.children):
        x = next((c for c in v.children if c.dep_ == "xcomp" and any(g.lower_ == "to" for g in c.children)), None)
        if x is not None: return predicate(x)
    if lem(v) == "be":
        p = next((c for c in v.children if c.dep_ in ("acomp", "attr")), None)
        if p is not None: return p
    return v

def clause_effect(item, v, frame):
    """Effect is a clause headed by v: N = its subject, V = its verb/predicate."""
    grp = "negated" if eff_neg(v) else "main"
    return rec(item, frame, subject_of(v), predicate(v), grp)

# No exemptions from the double-object rule (decided 29 Oct 2026): with two objects the second is scored for every verb,
# make, get, render and let included ('made him president' -> president, 'rendered the plan a failure' -> failure).
# make / get N + N lines are then left out of the headline results in scoring.prep (follow-up: version C).
DOUBLE_OBJ_EXCLUDE = set()

# ---- Rule switch: N + bare V as a misparsed double object (CAUSESENT_BARE_V_DOUBLE; on by default)
# 'causing people pain', 'cause students anguish': spaCy often tags the second noun as a bare verb (pain/VB) with the
# first noun as its subject. For every verb except those that take a bare infinitive (the studied make, let, help:
# 'made them leave', 'let the kids play', 'helped him move'; and, among the baseline verbs, perception verbs and the
# like: 'watched the kids play', 'heard her sing'), this is read as a double object and only the second word is scored;
# if it is unrated, the line gets no score.
# The frame 'V + N + bare V' also covers real clauses ('ensure they would be able', 'demand the government do
# anything'), so the rule applies only when the 'verb' looks like a noun: no auxiliary, no dependents of its own
# besides its subject and noun modifiers, and directly after its subject ('cause students severe anguish').
BARE_INF_VERBS = {"make", "let", "help", "see", "hear", "watch", "feel", "notice", "observe", "overhear", "have", "bid",
                  "sense", "spot", "witness"}
_NOMINAL_DEPS = {"det", "amod", "compound", "poss", "nummod", "quantmod"}
def _looks_like_noun(v, subj):
    if subj is None or subj.i > v.i: return False
    if any(c.dep_ not in _NOMINAL_DEPS and c.i != subj.i for c in v.children): return False
    return all(x.head.i == v.i and x.dep_ in _NOMINAL_DEPS for x in v.doc[subj.right_edge.i + 1: v.i])
def _bare_v_double(t, cf):
    if (config.BARE_V_DOUBLE and cf is not None and cf[0] == "V + N + bare V" and lem(t) not in BARE_INF_VERBS
            and _looks_like_noun(cf[2], cf[1])):
        return ("V + N + N (double object, bare-V parse)", cf[2], None, cf[3])
    return cf

def _adverbial_np(n):
    """'a lot', 'the most', 'a bit', 'the whole show': degree / duration phrases, not a second object."""
    if lem(n) in ("guy", "all", "both", "each"):   # 'you guys', 'us all': apposition, not a second object
        return True
    if lem(n) in ("lot", "bit", "most", "more", "less", "much", "little", "while", "deal", "iota") and not any(g.dep_ == "prep" and g.lower_ == "of" for g in n.children):
        return True
    if any(g.lower_ in ("every", "each") for g in n.children) and lem(n) in ("second", "minute", "hour", "day", "night", "week", "month", "year", "time"):
        return True    # 'every second', 'each time'
    if lem(n) == "way" and any(g.lower_ == "the" for g in n.children):
        return True    # 'the hard way'
    return any(g.lower_ == "whole" for g in n.children)   # 'the whole show', 'his whole life', 'the whole drive'

def _place_or_thing(n):
    """WordNet: the noun's first sense is a place, artifact, object, body part or substance ('truck', 'room', 'center')."""
    try:
        from nltk.corpus import wordnet as wn
        ss = wn.synsets(n.lower_, pos="n")[:1] or wn.synsets(lem(n), pos="n")[:1]
    except LookupError:
        return False
    return bool(ss) and ss[0].lexname() in ("noun.location", "noun.artifact", "noun.object", "noun.body", "noun.substance")
# verbs that push someone into something; not prompt / spark / trigger / deter, whose object is an event and whose
# into-phrase usually belongs to it ('prompted an investigation into the scare')
# ---- Rule switch CAUSESENT_SENSE_FILTERS (on; decided 1 Nov 2026): non-causal senses are left out (group 'other sense')
#   render        only with a result: 'rendered it useless', 'rendered the plan a failure' (not 'render a service / a
#                 verdict / an image / fat')
#   secure        not fastening or guarding: 'secure X to / with Y', 'secure X from Y', or an object that is a thing, place,
#                 body part or animal ('secure the door / the border / his ankles'); obtaining counts ('secure funding')
#   save          only rescue and prevention: 'save X from Y', or a person, group, animal, life, the world ... as object;
#                 not storing or keeping ('save the file', 'save money / time', 'save it for later'), not 'save you time'
#   contribute to not giving or taking part: an organisation, publication, discussion or fund as object ('contribute to a
#                 charity / the report / the discussion / a fund')
RENDER_FRAMES = {"V + N + ADJ", "V + N + V-ed", "V + N + N (double object)", "V + N + N (double object, two-object parse)",
                 "V + N + N (double object, clause parse)", "V + N + N (double object, bare-V parse)"}
GUARD_OBJECTS = {"server", "network", "data", "account", "website", "password", "perimeter", "internet", "wifi", "email",
                 "identity", "information", "connection", "computer", "database", "device", "router", "firewall"}
THING_LEX = {"noun.artifact", "noun.location", "noun.object", "noun.body", "noun.animal", "noun.substance"}
SAVE_KEEP_FRAMES = {"V + N + N (double object)", "V + N + N (double object, two-object parse)",
                    "V + N + N (double object, clause parse)", "V + N + N (double object, bare-V parse)"}
SAVE_OBJECTS = {"life", "lives", "world", "planet", "earth", "day", "job", "marriage", "relationship", "soul", "career",
                "company", "business", "species", "environment", "ass", "butt", "skin", "neck", "country", "nation", "city",
                "town", "village", "family", "child", "children", "baby", "people", "humanity", "mankind", "show", "season",
                "team", "game", "franchise", "film", "movie", "school", "industry", "economy", "democracy", "republic"}
PERSONAL = {"me", "us", "him", "her", "them", "you", "myself", "ourselves", "himself", "herself", "themselves", "yourself",
            "yourselves", "one", "everyone", "everybody", "someone", "somebody", "anyone", "anybody", "others"}
GIVE_TO = {"project", "fund", "charity", "campaign", "pot", "account", "ira", "401k", "pension", "plan", "effort",
           "cause", "kitty", "collection", "drive", "appeal", "foundation", "party", "magazine", "journal", "newspaper",
           "blog", "site", "website", "forum", "thread", "wiki", "anthology", "volume", "album", "book", "issue", "series"}
def _lexname(n):
    try:
        from nltk.corpus import wordnet as wn
        ss = wn.synsets(n.lower_, pos="n")[:1] or wn.synsets(lem(n), pos="n")[:1]
    except LookupError:
        return ""
    return ss[0].lexname() if ss else ""
# ---- 'determine' (added 3 Nov 2026, determine_plan_20261103.md): causal sense only ('the score determines your grade',
# 'what determines whether you get in'). NOT causal (group 'other sense'): an infinitive ('to determine whether',
# 'in order to determine'); no subject of its own (gerund after a preposition, imperative); a subject that is a person,
# an organisation or an agent pronoun (an agent decides: 'the judge / who / they determine(s) ...'); a subject that is an
# inquiry or measuring instrument (the test / study / analysis determines whether ...: epistemic). Passives are excluded
# as for every verb. A relative clause or participle takes the noun it modifies as subject ('factors that determine').
DET_AGENT_PRON = {"i", "you", "we", "they", "he", "she", "who", "whoever", "someone", "somebody", "one", "everyone", "everybody",
                  "anyone", "anybody", "people", "me", "us", "them", "him", "her", "whom", "those"}
DET_INSTRUMENT = {"test", "study", "analysis", "survey", "experiment", "investigation", "examination", "exam", "scan", "algorithm",
                  "software", "calculation", "measurement", "research", "report", "review", "audit", "inspection", "assessment",
                  "evaluation", "probe", "sensor", "instrument", "detector", "screening", "poll", "autopsy", "inquiry", "paper",
                  "article", "author", "authors", "investigator", "researcher", "scientist", "doctor", "police", "court", "judge",
                  "therapy", "therapist", "ambulance", "brain", "phone", "computer", "device", "app", "machine", "tool", "model",
                  "titre", "analyzer", "analyser", "program", "script", "function", "evidence", "data", "image", "images",
                  "leadership", "admin", "admins", "moderator", "moderators", "management", "committee", "board", "government"}
DET_EPISTEMIC_CONJ = {"look", "observe", "check", "test", "examine", "find", "see", "measure", "analyze", "analyse", "calculate",
                      "read", "ask", "search", "scan", "investigate", "inspect", "evaluate", "assess", "study", "survey", "run"}
DET_ACL_HEADS = {"time", "way", "ways", "ability", "method", "process", "tool", "means", "step", "steps", "trouble", "difficulty",
                 "problem", "challenge", "task", "job", "key", "goal", "purpose", "technique", "approach", "effort", "attempt"}
def _det_subject(t):
    s = next((c for c in t.children if c.dep_ in ("nsubj", "csubj")), None)
    h = t
    while s is None and h.dep_ == "conj" and h.head is not h:
        h = h.head; s = next((c for c in h.children if c.dep_ in ("nsubj", "csubj")), None)
    return s
def _agent_noun(n):
    """person or organisation: any of the first three WordNet senses (the first sense of 'group' is a Tops class)."""
    try:
        from nltk.corpus import wordnet as wn
        ss = (wn.synsets(n.lower_, pos="n") or wn.synsets(lem(n), pos="n"))[:3]
    except LookupError:
        return False
    return any(s.lexname() in ("noun.person", "noun.group") for s in ss)
def determine_other_sense(t, frame=None):
    """True if this use of 'determine' is not the causal sense (see the comment above)."""
    # 'determine that [clause]' is epistemic (find out, conclude) in about 99% of the lines (AskReddit Jan 2018: 383 lines,
    # 3 causal), so a clause with an explicit 'that' is left out; whether / if / wh-clauses are kept.
    if frame == "V + that-clause" and any(c.dep_ == "ccomp" and c.i > t.i and any(g.dep_ == "mark" and g.lower_ == "that" for g in c.children)
                                          for c in t.children): return True
    if any(c.dep_ == "aux" and c.lower_ == "to" for c in t.children): return True          # infinitive
    if any(c.dep_ == "aux" and c.lower_ in ("can", "could", "cannot", "ca") for c in t.children): return True   # ability: 'can determine'
    if t.lower_ == "determined" and any(c.dep_ == "xcomp" and any(g.dep_ == "aux" and g.lower_ == "to" for g in c.children) for c in t.children): return True   # 'determined to V' = resolved
    h = t
    while h.dep_ == "conj" and h.head is not h:                                               # 'look at X and determine whether'
        if lem(h.head) in DET_EPISTEMIC_CONJ: return True
        h = h.head
    if t.dep_ == "acl" and (t.tag_ == "VB" or (t.tag_ == "VBG" and lem(t.head) in DET_ACL_HEADS)): return True   # imperative parsed as acl ('Step 2: Determine ...'); 'a hard time determining'
    own = next((c for c in t.children if c.dep_ in ("nsubj", "csubj")), None)
    if own is None and t.dep_ in ("relcl", "acl") and t.head.pos_ in ("NOUN", "PROPN", "PRON"): subj = t.head
    else: subj = _det_subject(t)
    if subj is None: return True
    if subj.dep_ == "csubj": return False
    if subj.pos_ == "PRON": return subj.lower_ in DET_AGENT_PRON
    if lem(subj) in DET_INSTRUMENT or subj.lower_ in DET_INSTRUMENT: return True
    if subj.pos_ == "PROPN" or subj.ent_type_ in ("PERSON", "ORG", "NORP"): return True
    return _agent_noun(subj)
def other_sense(item, t, frame, N):
    """True if this use of render / secure / save / contribute to is not the causal sense (switch SENSE_FILTERS)."""
    if not config.SENSE_FILTERS: return False
    if item == "determine": return determine_other_sense(t, frame)
    preps = {c.lower_ for c in t.children if c.dep_ in ("prep", "dative")}
    if item == "render": return frame not in RENDER_FRAMES
    if item == "secure":
        if preps & {"to", "with", "from", "against", "onto", "into", "in"}: return True
        if N is None: return True
        # added 2 Nov 2026 (COCA spot check: 'secured your server', 'secure a certain network'): computer-security
        # objects that WordNet does not class as things. ('to / with' on the object was tested and dropped: mostly
        # obtaining, 'secure a contract with', 'secure access to'.)
        if lem(N) in GUARD_OBJECTS or N.lower_ in GUARD_OBJECTS: return True
        if N.pos_ == "PROPN" or N.ent_type_ in ("GPE", "LOC", "FAC"): return True
        return N.pos_ in ("PRON",) or _lexname(N) in THING_LEX
    if item == "save":
        if "from" in preps: return False
        if frame in SAVE_KEEP_FRAMES: return True
        if N is None: return True
        if N.lower_ in PERSONAL or lem(N) in SAVE_OBJECTS or N.lower_ in SAVE_OBJECTS: return False
        if N.pos_ == "PROPN": return False
        return _lexname(N) not in ("noun.person", "noun.group", "noun.animal")
    if item == "contribute to":
        if N is None or N.pos_ in ("PRON", "VERB") or N.tag_ == "VBG": return False
        return lem(N) in GIVE_TO or _lexname(N) in ("noun.group", "noun.communication", "noun.possession")
    return False

INTO_VERBS = {"force", "compel", "persuade", "convince", "tempt", "provoke", "induce", "encourage", "inspire", "motivate",
              "stimulate", "oblige"}
def complement_of(t, item=None):
    """Returns (frame, N, V, negated) for a verb-like trigger t, or None."""
    kids = list(t.children)
    dobj = next((c for c in kids if c.dep_ in ("dobj", "dative") and c.i > t.i), None)
    # ---- Rule switch: double objects (CAUSESENT_DOUBLE_OBJECT; on by default)
    # 'caused me concerns' -> score the second noun (the thing caused), not the indirect object. For every verb (no
    # exemptions): 'made him president' -> president, 'render it a loser' -> loser.
    if config.DOUBLE_OBJECT and lem(t) not in DOUBLE_OBJ_EXCLUDE:
        d2 = next((c for c in kids if c.dep_ == "dobj" and c.i > t.i), None)
        if (dobj is not None and dobj.dep_ == "dative" and dobj.pos_ in ("PRON", "PROPN", "NOUN") and d2 is not None
                and d2.i > dobj.i and not _adverbial_np(d2)):
            return ("V + N + N (double object)", d2, None, False)
        # spaCy's third parse: two direct objects ('made the kitchen a mess': kitchen/dobj, mess/dobj). This parse is
        # also given to hyphenated words split by the tokenizer ('re - election'), appositions ('help you gentlemen'),
        # lists and duration phrases, so it counts only when the two nouns are adjacent, the first does not end in a
        # comma or conjunction, and the second is a common noun opening with a determiner, quantifier or adjective
        # ('guaranteeing same-sex couples the right to marry', 'secured her a pardon', 'rendering Scott a quadriplegic').
        dobjs = [c for c in kids if c.dep_ == "dobj" and c.i > t.i and c.pos_ in ("PRON", "PROPN", "NOUN")]
        if len(dobjs) >= 2:
            d1, d2 = dobjs[0], dobjs[1]
            if (d2.left_edge.i == d1.right_edge.i + 1 and d2.left_edge.i < d2.i and d2.is_alpha
                    and d2.tag_ in ("NN", "NNS") and t.doc[d1.right_edge.i].dep_ not in ("cc", "punct")
                    and t.doc[d2.left_edge.i].tag_ in ("DT", "PDT", "PRP$", "CD", "JJ", "JJR", "JJS", "RB")
                    and not _adverbial_np(d2)):
                return ("V + N + N (double object, two-object parse)", d2, None, False)
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
        neg = eff_neg(c)
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
            return (("V + N + to V" if N is not None else "V + to V"), N, predicate(c), neg)
        if c.tag_ == "VBN":
            return ("V + N + V-ed", N, c, neg) if N is not None else None
        if c.tag_ == "VB" and N is not None:
            return ("V + N + bare V", N, c, neg)
    # from / for / in + V-ing ; with / for + N
    preps = [c for c in kids if c.dep_ == "prep"] + ([c for c in dobj.children if c.dep_ == "prep" and c.lower_ in ("with", "for", "from")] if dobj is not None else [])
    # 'forced him into resigning', 'provoked them into a war' (decided 1 Nov 2026): the into-phrase is the effect, for the
    # verbs of inducing (not 'helped her into the car')
    # (only an into-phrase attached to the verb itself, not one inside the object ('expansion into space'), and not a place
    # or thing moved into ('forced them into his truck'))
    if lem(t) in INTO_VERBS and dobj is not None and lem(dobj) != "way":     # not 'forced their way into'
        for c in kids:
            if c.dep_ != "prep" or c.lower_ != "into" or c.i < dobj.i: continue
            g = next((x for x in c.children if x.dep_ in ("pcomp", "pobj")), None)
            if g is None: continue
            if g.tag_ == "VBG": return ("V + N + into V-ing", dobj, g, eff_neg(g))
            if g.pos_ == "NOUN" and not _place_or_thing(g): return ("V + N + into N", g, None, False)
    for c in preps:
        if c.lower_ in ("from", "for", "in", "with"):
            g = next((x for x in c.children if x.dep_ in ("pcomp", "pobj")), None)
            if g is None: continue
            if g.tag_ == "VBG" and c.lower_ in ("from", "for", "in"):
                return (f"V + N + {c.lower_} V-ing" if dobj is not None else f"V + {c.lower_} V-ing", dobj, g, eff_neg(g))
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

# ---- Rule switches CAUSESENT_PARSE_FIXES (on) and CAUSESENT_REATTACH (off) (1 Nov 2026): parser repairs
COPULAS = {"be", "seem", "appear", "become", "remain", "look", "get"}
def _adjectival_passive(t):
    """'is pretty much guaranteed to happen', 'seems guaranteed to fail': a participle that spaCy attaches to a copula as
    an adjective (acomp / xcomp / oprd) is a passive."""
    return t.tag_ == "VBN" and t.dep_ in ("acomp", "xcomp", "oprd") and t.head is not t and lem(t.head) in COPULAS
def _attributive_participle(t):
    """'experimentally induced hepatitis', 'a professionally produced video': a past participle before its noun (amod /
    compound) is a passive, like the participle after it ('the damage caused by', acl)."""
    return t.tag_ == "VBN" and t.dep_ in ("amod", "compound") and t.head.pos_ in ("NOUN", "PROPN") and t.head.i > t.i
def _adjacent_noun(t):
    """A noun that directly follows the verb (after determiners, adjectives, numbers and noun compounds) but that the
    parser attached elsewhere: 'alcohol causes liver damage, blood sugar problems, ...' (damage parsed as the subject of
    a later verb). Only when no punctuation or verb intervenes."""
    doc = t.doc
    if t.dep_ in ("amod", "compound", "acl", "nmod", "npadvmod", "advmod") or t.tag_ not in ("VB", "VBD", "VBZ", "VBP", "VBG"):
        return None       # an adjective-like participle ('a motivating pace', 'enabling doctors'), not a verb with an object
    if t.tag_ == "VBG" and t.dep_ not in ("ROOT", "conj", "xcomp", "ccomp", "advcl", "pcomp", "relcl"): return None
    for k in range(t.i + 1, min(len(doc), t.i + 7)):
        x = doc[k]
        if x.is_punct or x.pos_ in ("VERB", "AUX", "ADP", "SCONJ", "CCONJ", "PART", "PRON"): return None
        if x.pos_ in ("DET", "ADJ", "NUM", "ADV") or x.dep_ in ("compound", "amod", "nummod", "det", "poss"): continue
        if x.pos_ in ("NOUN", "PROPN"):
            return x if x.head is not t else None
        return None
    return None

def _subject(t):
    """Subject of the causal verb t: its own subject, the antecedent of a relative pronoun ('a skill that gets me
    raises' -> skill), the subject of the first conjunct ('went and got me a job'), '(imperative)', or None."""
    s = subject_of(t)
    if s is not None and lem(s) in ("that", "which", "who") and t.dep_ == "relcl": s = t.head
    if s is None and t.dep_ == "conj": s = subject_of(t.head)
    if (s is None and t.dep_ == "ROOT" and t.tag_ == "VB"
            and all(lem(c) == "do" for c in t.children if c.dep_ in ("aux", "auxpass"))):   # 'Get me ...', "Don't let ..."
        return "(imperative)"
    return s

def _verb_tags(r, item, t):
    s = _subject(t)
    r["subj_words"] = s if isinstance(s, str) else (lem(s) if s is not None else "")
    r["subj_pos"] = "PRON" if isinstance(s, str) else (s.pos_ if s is not None else "")
    if item == "cause":
        cr = cause_reading(t.doc, t.i)
        r["cause_reading"] = cr if cr is not None else ""
    return r

# =====================================================================================================================
# ---- The 21 items added on 2 Nov 2026 (word_list_v6.tsv; decisions in query_checklist.txt, 1-2 Nov 2026)
# Sense rules for so / since / as / for / therefore / thus / hence / consequently: causesent/senses.py (tested 1 Nov
# 2026 in sense_check_20261101/). Lines in a non-causal sense go to group 'other sense' (out of the results); lines
# that belong to another studied expression go to group 'overlap' (out; counted once, for that expression):
#   'as a result of' in a 'result of' sample, 'as a consequence of' in a 'consequence of' sample, 'the reason why' in a
#   'why' sample, 'so as to' in a 'so' sample.
# Negative forms (approved by Dean 2 Nov 2026, negative_forms_draft_20261102.md): the claim of cause / reason /
# explanation is denied ('X doesn't explain Y', 'that's not why Y', 'not on account of X', 'X is not the cause of Y',
# 'nobody is responsible for Y', 'there is no reason why Y'); so / therefore / thus / hence / consequently have none;
# negation inside the effect stays group 'negated'; 'no' / 'zero' on the effect noun is a negative form (as before).
NOT_WORDS = {"not", "n't", "never", "nt"}
def _not_before(doc, i):
    """'not' (or n't / never) directly before position i ('not since', 'not on account of', 'not in order to')."""
    return i > 0 and doc[i - 1].lower_ in NOT_WORDS
def _verbish(v): return v is not None and v.pos_ in ("VERB", "AUX", "ADJ", "NOUN", "PROPN")

def _adv_item(doc, item, t):
    """so / therefore / thus / hence / consequently: the effect is the clause they introduce ('X, so Y'; 'X. Therefore Y';
    'X and thus Y'; 'so that Y' (purpose, unified with 'so' 1 Nov 2026); 'thus causing delays' -> the participle and its
    object; 'hence the name' -> the noun)."""
    label, why = causal_sense(doc, t.i, t.i + 1)
    if label != "causal":
        r = rec(item, "other sense", group="overlap" if why.startswith("so as to") else "other sense"); r["sense"] = why
        return r
    nx = doc[t.i + 1] if t.i + 1 < len(doc) else None
    if item == "so" and nx is not None and nx.lower_ == "that":
        v = nx.head if nx.head is not nx else None
        if not _verbish(v) or v.i < nx.i: return rec(item, "no effect")
        r = clause_effect(item, v, "so that + clause")
    else:
        if t.dep_ == "cc":                                    # 'X, so Y' parsed as coordination: Y = the next conjunct
            v = next((c for c in t.head.children if c.dep_ == "conj" and c.i > t.i), None)
        else:
            v = t.head if t.head is not t else None
        if item == "so" and v is not None and v.i < t.i:      # attached to the first clause: take the clause after 'so'
            v = next((c for c in v.children if c.i > t.i and c.dep_ in ("conj", "ccomp", "advcl", "parataxis")), v)
            if v.i < t.i: return rec(item, "no effect")
        if v is None: return rec(item, "no effect")
        if v.pos_ in ("NOUN", "PROPN") and item == "hence":   # 'hence the name'
            r = rec(item, "hence + N", v)
        elif v.tag_ == "VBG" and subject_of(v) is None and v.i > t.i:   # 'thus causing delays'
            obj = next((c for c in v.children if c.dep_ in ("dobj", "attr")), None)
            r = rec(item, "ADV + V-ing", obj, v, "negated" if eff_neg(v) else "main")
        elif _verbish(v):
            r = clause_effect(item, v, "ADV + clause")
        else:
            return rec(item, "no effect")
    r["sense"] = why; r["polarity"] = "positive"           # these adverbs have no negative forms (decided 2 Nov 2026)
    return r

def _main_clause_of(t):
    """For a subordinator t ('since', 'as', 'for', 'that' of 'given that'): the verb of the main clause, or None.
    'Y since X' / 'Since X, Y': the reason clause's verb is t's head; the main clause is that verb's head."""
    rv = t.head
    if rv is t: return None
    if rv.i < t.i:                         # t attached to the main clause itself (e.g. 'for' parsed as cc / prep)
        return rv if _verbish(rv) else None
    m = rv.head
    if m is rv: return None
    return m if _verbish(m) else None

def _sub_item(doc, item, t):
    """since / as / for (causal senses only, senses.py) and given that: the effect is the main clause."""
    if item == "given that":
        nx = doc[t.i + 1] if t.i + 1 < len(doc) else None
        if nx is None or nx.lower_ != "that": return None
        if any(c.dep_ in ("aux", "auxpass", "nsubjpass") for c in t.children) or nx.dep_ != "mark" or \
                (t.i > 0 and doc[t.i - 1].lower_ in ("a", "the")):          # 'it is a given that'
            return rec(item, "other sense", group="other sense")      # 'evidence was given that ...', 'given that choice'
        if t.dep_ == "ROOT": return rec(item, "no effect")             # no main clause in the line (cut-off context)
        rv = nx.head if nx.head is not nx else None
        if rv is None: return rec(item, "no effect")
        m = next((a for a in rv.ancestors if a.i != t.i and a.i not in range(t.i, nx.i + 1)), None)
        if m is not None and m.i > t.i and m.i in {x.i for x in rv.subtree}: m = None
        if not _verbish(m) or m.i == rv.i: return rec(item, "no effect")
        r = clause_effect(item, m, "effect on the left (clause)")
    else:
        if item in ("as", "for"):      # outside their frames these words are not candidates at all, as in the narrowed queries
            pv = doc[t.i - 1] if t.i > 0 else None
            first = t.i == t.sent.start or all(x.is_punct for x in doc[t.sent.start:t.i])
            if not ((pv is not None and pv.text == ",") or (item == "as" and first)): return None
        label, why = causal_sense(doc, t.i, t.i + 1)
        if label != "causal":
            r = rec(item, "other sense", group="other sense"); r["sense"] = why; return r
        m = _main_clause_of(t)
        if m is None: return rec(item, "no effect")
        r = clause_effect(item, m, "effect on the left (clause)"); r["sense"] = why
    r["polarity"] = polarity(_not_before(doc, t.i))
    return r

# literal 'in the light of the moon / the fire / the screen' (decided 2 Nov 2026: left out, group 'other sense')
LIGHT_LITERAL = {"moon", "sun", "fire", "lamp", "candle", "candles", "day", "dawn", "morning", "evening", "sunset",
                 "sunrise", "torch", "flashlight", "lantern", "lanterns", "street", "streetlight", "streetlights", "window",
                 "screen", "star", "stars", "headlight", "headlights", "fireplace", "flame", "flames", "hearth", "bonfire",
                 "campfire", "moonlight", "sunlight", "twilight", "dusk", "night", "lightning", "lighthouse", "tv",
                 "television", "monitor", "computer", "phone", "match", "lamps", "bulb", "candlelight", "firelight"}
def _literal_light(N):
    if N is None: return False
    return lem(N) in LIGHT_LITERAL or N.lower_ in LIGHT_LITERAL or _lexname(N) in ("noun.object", "noun.artifact")

def _noun_neg(t):
    """Negative form of a noun item: 'no reason', 'X is not the cause of Y', 'the cause of Y is not X', 'nothing is the
    result of Y'."""
    if is_neg(t): return True
    h = t.head
    if h is not t and lem(h) in COPULAS and t.dep_ in ("attr", "acomp", "nsubj", "nsubjpass", "ccomp"):
        return is_neg(h) or (t.dep_ in ("attr", "acomp") and neg_subject(h))
    return False

def _named_effect(item, t):
    """'X is the result / consequence / effect of Y' -> X; 'the effect of Y was X' -> X (X a noun); 'the result of Y is that
    X' -> the clause X."""
    h = t.head
    if h is not t and lem(h) in COPULAS and t.dep_ in ("attr", "acomp"):
        X = subject_of(h)
        if X is not None and X.lower_ in ("which", "that", "who") and h.dep_ == "relcl": X = h.head
        if X is None and h.dep_ in ("relcl", "acl") and h.head.pos_ in ("NOUN", "PROPN"): X = h.head
        if X is None: return rec(item, "no effect")
        if X.pos_ in ("VERB", "AUX") or X.dep_ == "csubj":         # 'what happened was the result of'
            return clause_effect(item, X, "X is the N of Y")
        return rec(item, "X is the N of Y", X)
    if h is not t and lem(h) in COPULAS and t.dep_ in ("nsubj", "nsubjpass"):
        # only a NOUN names the effect ('the effect of the storm was damage'); an adjective says how it was
        # ('the effect was significant / positive', 'the consequences are grave'): not counted
        a = next((c for c in h.children if c.dep_ == "attr" and c.i > h.i and c.pos_ in ("NOUN", "PROPN")), None)
        if a is not None: return rec(item, "the N of Y is X", a)
        cc = next((c for c in h.children if c.dep_ == "ccomp" and c.i > h.i), None)
        if cc is not None and any(g.lower_ == "that" and g.dep_ == "mark" for g in cc.children):
            return clause_effect(item, cc, "the N of Y is that X")
        return rec(item, "no effect")
    return rec(item, "no effect")       # appositions ('X, the result of Y') left out: mostly lists and misparses

def _noun_item(doc, item, t):
    """the cause of / the effect of / result of / consequence of / reason (nouns)."""
    if t.pos_ not in ("NOUN", "PROPN") and not (item == "the cause of" and t.lower_ in ("cause", "causes")): return None
    nx = doc[t.i + 1] if t.i + 1 < len(doc) else None
    if item == "reason":
        if lem(t) != "reason" and t.lower_ not in ("reason", "reasons"): return None
        p = next((c for c in t.children if c.dep_ == "prep" and c.lower_ == "for"), None)
        cl = next((c for c in t.children if c.dep_ in ("relcl", "acl") and c.i > t.i and c.pos_ in ("VERB", "AUX")
                   and not (c.tag_ == "VBN" and subject_of(c) is None)), None)     # not 'reasons painted on cars'
        if cl is not None and any(g.lower_ == "to" and g.dep_ == "aux" for g in cl.children):
            r = rec(item, "reason to V (not counted)", group="no effect")       # 'no reason to worry': not an explained Y
        elif cl is not None:
            r = clause_effect(item, cl, "the reason (why/that) Y")
        elif p is not None:
            po = prep_object(p)
            if po is None: r = rec(item, "no effect")
            elif po[0] == "noun": r = rec(item, "the reason for N", po[1])
            else: r = rec(item, "the reason for V-ing", po[1], po[2], "negated" if eff_neg(po[2]) else "main")
        else:
            r = rec(item, "no effect")                                       # 'for some reason', 'the reason is ...'
        r["polarity"] = polarity(_noun_neg(t)); return r
    if nx is None or nx.lower_ != "of": return None
    if item in ("result of", "consequence of"):
        k = t.i - 1
        while k >= 0 and doc[k].tag_ in ("JJ", "JJR", "JJS", "RB") and k > t.i - 4: k -= 1   # 'as a direct result of'
        if k >= 1 and doc[k].lower_ in ("a", "an") and doc[k - 1].lower_ == "as":
            return rec(item, "as a ... of (counts for 'as a result (of)' / 'as a consequence (of)')", group="overlap")
        r = _named_effect(item, t)
    elif item == "the effect of X on Y":
        r = _named_effect(item, t)          # only lines that name the effect (decided 1 Nov 2026); 'on Y' is not scored
    else:                                   # the cause of: the noun after 'of' (no filter, 1 Nov 2026)
        po = prep_object(nx)
        if po is None: r = rec(item, "no effect")
        elif po[0] == "noun": r = rec(item, "the cause of N", po[1])
        else: r = rec(item, "the cause of V-ing", po[1], po[2], "negated" if eff_neg(po[2]) else "main")
    r["polarity"] = polarity(_noun_neg(t)); return r

def _why_item(doc, item, t):
    """why: 'that's / this is / which is why Y' and questions ('why did Y?', 'I wonder why Y'; included 1 Nov 2026):
    the clause after 'why'. 'the reason why Y' counts for 'reason' (group 'overlap')."""
    if t.lower_ != "why": return None
    pv = doc[t.i - 1] if t.i > 0 else None
    if (pv is not None and pv.lower_ in ("reason", "reasons")) or (t.head.dep_ in ("relcl", "acl") and lem(t.head.head) == "reason"):
        return rec(item, "the reason why (counts for 'reason')", group="overlap")
    v = t.head if t.head is not t else None
    if v is None or v.i < t.i or not _verbish(v): return rec(item, "no effect")
    h = v.head
    frame = "that's why + clause" if (h is not v and lem(h) == "be" and h.i < t.i) else "why + clause"
    r = clause_effect(item, v, frame)
    r["polarity"] = polarity(h is not v and lem(h) == "be" and h.i < t.i and is_neg(h))     # "that's not why Y"
    return r

AVOID_VERBS = {"avoid", "prevent", "stop"}
def _purpose_item(doc, item, t):
    """in order to V (N) / in order that X / in order for X to V / so as to V: V = the verb, N = its object (or the
    subject of the purpose clause). The to-clause is tagged 'avoid_V' when its verb is avoid / prevent / stop (a tag,
    like the prevention words)."""
    nx = doc[t.i + 1] if t.i + 1 < len(doc) else None
    if nx is not None and nx.lower_ in NOT_WORDS and t.i + 2 < len(doc): nx = doc[t.i + 2]   # 'in order not to', 'so as not to'
    if nx is None: return None
    if t.lower_ == "order":
        if t.i == 0 or doc[t.i - 1].lower_ != "in" or nx.lower_ not in ("to", "that", "for"): return None
        anchor = doc[t.i - 1]
    elif t.lower_ == "as":
        if t.i == 0 or doc[t.i - 1].lower_ != "so" or nx.lower_ != "to": return None
        anchor = doc[t.i - 1]
    else:
        return None
    if nx.lower_ == "that":
        v = nx.head if nx.head is not nx else None
        if not _verbish(v) or v.i < nx.i: return rec(item, "no effect")
        r = clause_effect(item, v, "in order that X")
    else:
        to = nx if nx.lower_ == "to" else next((x for x in doc[nx.i + 1:nx.i + 8] if x.lower_ == "to"), None)
        if to is None: return rec(item, "no effect")
        V = to.head if to.dep_ == "aux" and to.head.i > to.i else next((x for x in doc[to.i + 1:to.i + 4] if x.pos_ in ("VERB", "AUX")), None)
        if V is None: return rec(item, "no effect")
        if nx.lower_ == "for":
            N = subject_of(V) or next((x for x in doc[nx.i + 1:to.i] if x.pos_ in ("NOUN", "PROPN", "PRON")), None)
            r = rec(item, "in order for N to V", N, V, "negated" if eff_neg(V) else "main")
        else:
            N = next((c for c in V.children if c.dep_ in ("dobj", "attr")), None)
            frame = ("so as to V" if t.lower_ == "as" else "in order to V") + (" + N" if N is not None else "")
            r = rec(item, frame, N, predicate(V) if lem(V) == "be" else V, "negated" if eff_neg(V) else "main")
        r["avoid_V"] = lem(V) in AVOID_VERBS
    r["polarity"] = polarity(_not_before(doc, anchor.i))       # 'not in order to win'
    return r

# ---- Noun baseline (decided 1 Nov 2026): every other noun in 'the N of X' (X scored) and 'X is the N of Y' (X scored) in
# the baseline samples; generic item 'other nouns'. Left out: the tested nouns (targets.NOUN_BASE_EXCLUDE) and the
# quantity / part / kind words, which are looked through elsewhere ('a lot of X', 'part of X').
def _noun_base(doc, item, t):
    if t.pos_ != "NOUN" or lem(t) in NOUN_BASE_EXCLUDE or lem(t) in QUANT_DROP or lem(t) in QUANT_OF: return None
    of = next((c for c in t.children if c.dep_ == "prep" and c.lower_ == "of" and c.i == t.i + 1), None)
    if of is None: return None
    h = t.head
    if h is not t and lem(h) == "be" and t.dep_ == "attr":
        X = subject_of(h)
        if X is None: return None
        if X.lower_ in ("which", "that", "who") and h.dep_ == "relcl": X = h.head
        r = rec(item, "X is the N of Y", X)
    else:
        po = prep_object(of)
        if po is None: return None
        r = rec(item, "the N of X", po[1]) if po[0] == "noun" else rec(item, "the N of V-ing", po[1], po[2], "negated" if eff_neg(po[2]) else "main")
    r["noun"] = lem(t); r["polarity"] = polarity(_noun_neg(t))
    return r
# =====================================================================================================================

NEW_CONN = {"as a result (of)", "as a consequence (of)", "on account of", "in light of"}
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
            if "polarity" not in r:   # connectives set their own polarity (see 'conn' below)
                r["polarity"] = "positive" if kind in ("conn", "sconj") else polarity(word_neg(t))
            if kind == "conn" and item in NEW_CONN and r.get("_not_before"): r["polarity"] = "negative"   # 'not on account of X'
            r.pop("_not_before", None)
            if r.get("effect_no") and kind != "sconj" and r.get("group") in ("main", "negated", "passive"):
                r["polarity"] = "negative"     # 'caused no damage'
            out.append(r)
    return out

def _one(doc, item, lemma, kind, t):
    kids = list(t.children)
    if kind == "adv": return _adv_item(doc, item, t) if t.lower_ == item else None
    if kind == "sub":
        w = "given" if item == "given that" else item
        return _sub_item(doc, item, t) if t.lower_ == w else None
    if kind == "noun": return _noun_item(doc, item, t)
    if kind == "why": return _why_item(doc, item, t)
    if kind == "purpose": return _purpose_item(doc, item, t)
    if kind == "noun_base": return _noun_base(doc, item, t)
    if kind == "verb":
        if t.pos_ not in ("VERB", "AUX") and not t.tag_.startswith("VB"): return None
        if item == "make":   # 'make sure' and 'make a difference to' are separate items
            if any(c.lower_ == "sure" for c in kids): return None
            if any(c.dep_ == "dobj" and c.lemma_.lower() == "difference" for c in kids): return None
        if item == "require" and any(c.dep_ == "prep" and c.lower_ == "for" for c in kids) and t.tag_ == "VBN": return None
        passive = any(c.dep_ in ("nsubjpass", "auxpass", "agent") for c in kids) or (t.dep_ == "acl" and t.tag_ == "VBN")
        if not passive and config.PARSE_FIXES and (_adjectival_passive(t) or _attributive_participle(t)):
            passive = True        # 'is pretty much guaranteed to', 'experimentally induced hepatitis'
        if passive:
            N = next((c for c in kids if c.dep_ == "nsubjpass"), None)
            if N is None and _adjectival_passive(t): N = next((c for c in t.head.children if c.dep_ in ("nsubj", "nsubjpass")), None)
            if N is None and (t.dep_ == "acl" or _attributive_participle(t)): N = t.head
            comp = next((c for c in kids if c.dep_ in ("xcomp", "ccomp") and c.i > t.i), None)
            return rec(item, "passive", N, comp, "passive")
        cf = complement_of(t, item)
        if cf is None:
            if t.dep_ == "relcl" and t.head.pos_ in ("NOUN", "PROPN"):   # 'the damage they cause': object relativised
                r = rec(item, "V + N", t.head,
                        group="other sense" if item in ("render", "secure", "save", "determine") and other_sense(item, t, "V + N", t.head) else "main")
            elif config.REATTACH and _adjacent_noun(t) is not None:     # 'causes liver damage, ...' (switch off by default)
                r = rec(item, "V + N (reattached)", _adjacent_noun(t))
            else:
                r = rec(item, "no effect")
            return _verb_tags(r, item, t)
        # colloquial 'cause = because, judged on spaCy's own frame (before the bare-V rule below)
        if item == "cause" and config.CAUSE_FILTER and is_because_line(t, cf[0]):
            return _verb_tags(rec(item, cf[0], cf[1], cf[2], "because"), item, t)
        cf = _bare_v_double(t, cf)
        frame, N, V, neg = cf[:4]
        if item in ("render", "secure", "save", "determine") and other_sense(item, t, frame, N):
            return _verb_tags(rec(item, frame, N, V, "other sense"), item, t)
        r = rec(item, frame, N, V, "negated" if neg else "main")
        if len(cf) > 4:   # with/for-N frame: also keep the direct object's score
            ot = _nopron(conj_group(cf[4])); r["obj_words"] = " ".join(lem(x) for x in ot); r["score_obj"] = score(ot)
            r["obj_mods"] = " ".join(lem(x) for x in mods_of(ot))
            r["obj_forms"] = forms(ot); r["obj_mods_forms"] = forms(mods_of(ot))
        return _verb_tags(r, item, t)
    if kind == "impl":
        if not t.tag_.startswith("VB"): return None
        if item in ("manage to", "fail to"):
            c = next((c for c in kids if c.dep_ in ("xcomp",) and any(g.lower_ == "to" for g in c.children)), None)
            if c is None: return None
            return rec(item, "V + to V", None, predicate(c), "negated" if eff_neg(c) else "main")
        p = next((c for c in kids if c.dep_ == "prep" and c.lower_ == "in"), None)
        if p is None: return None
        po = prep_object(p)
        if po is None: return None
        if po[0] == "noun": return rec(item, "V + PREP + N", po[1])
        return rec(item, "V + PREP + V-ing", po[1], po[2], "negated" if eff_neg(po[2]) else "main")
    if kind == "phrasal":
        if not (t.tag_.startswith("VB") or t.pos_ in ("VERB", "AUX")): return None
        if item == "make sure":
            sure = next((c for c in kids if c.lower_ == "sure"), None)
            if sure is None: return None
            c = next((g for g in list(sure.children) + kids if g.dep_ in ("ccomp", "xcomp") and g.i > sure.i), None)
            if c is None: return rec(item, "no effect")
            if any(g.lower_ == "to" and g.dep_ == "aux" for g in c.children):
                return rec(item, "V + to V", None, predicate(c), "negated" if eff_neg(c) else "main")
            return rec(item, "V + that-clause", subject_of(c), predicate(c), "negated" if eff_neg(c) else "main")
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
            frame, N, V, neg = _bare_v_double(t, cf)[:4]
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
            # 'did not result from': a negative form, kept (CAUSESENT_NEG_RULES, 1 Nov 2026); before: left out
            return rec(item, "effect on the left", N, None, "negated" if not config.NEG_RULES and is_neg(t) else "main")
        if any(c.dep_ in ("nsubjpass", "auxpass") for c in kids): return rec(item, "passive", subject_of(t), None, "passive")
        po = prep_object(p)
        if po is None: return rec(item, "no effect")
        if po[0] == "noun":
            if item == "contribute to" and other_sense(item, t, "V + PREP + N", po[1]):
                return rec(item, "V + PREP + N", po[1], None, "other sense")      # 'contributed to a fund'
            return rec(item, "V + PREP + N", po[1])
        return rec(item, "V + PREP + V-ing", po[1], po[2], "negated" if eff_neg(po[2]) else "main")
    if kind == "sconj":    # baseline for the connectives: any other subordinating conjunction, effect = its main clause
        if t.pos_ != "SCONJ" or t.lower_ in SCONJ_EXCLUDE or lem(t) in SCONJ_EXCLUDE: return None
        if t.lower_ in SCONJ_EXCLUDE_AFTER and t.i > 0 and doc[t.i - 1].lower_ in SCONJ_EXCLUDE_AFTER[t.lower_]: return None
        sub = t.head
        main = sub.head if sub.dep_ in ("advcl", "ccomp", "conj") and sub.head is not sub else None
        if main is None or main.pos_ not in ("VERB", "AUX", "ADJ", "NOUN"): return None
        r = clause_effect(item, main, "effect on the left (clause)"); r["conj"] = lem(t)
        return r
    if kind == "conn":
        if item == "because (clause)":
            if t.lower_ != "because" or t.dep_ != "mark": return None
            sub = t.head                                # head of the because-clause
            main = sub.head if sub.dep_ in ("advcl", "ccomp", "conj") and sub.head is not sub else None
            if main is None or main.pos_ not in ("VERB", "AUX", "ADJ", "NOUN"): return None
            return clause_effect(item, main, "effect on the left (clause)")
        words = {"because of": ["because"], "due to": ["due"], "thanks to": ["thanks"], "owing to": ["owing"],
                 "as a result (of)": ["result"], "as a consequence (of)": ["consequence"], "on account of": ["account"],
                 "in light of": ["light"]}[item]
        if t.lower_ not in words: return None
        nxt = doc[t.i + 1] if t.i + 1 < len(doc) else None
        need = {"because of": "of", "due to": "to", "thanks to": "to", "owing to": "to", "as a result (of)": None,
                "as a consequence (of)": None, "on account of": "of", "in light of": "of"}[item]
        if need is not None and (nxt is None or nxt.lower_ != need): return None
        new_item = item in ("as a result (of)", "as a consequence (of)", "on account of", "in light of")
        if item in ("as a result (of)", "as a consequence (of)"):
            # 'as a result(,)', 'as a result of N', 'as a direct result of N' (adjective slot, decided 2 Nov 2026)
            k = t.i - 1
            while k >= 0 and doc[k].tag_ in ("JJ", "JJR", "JJS", "RB") and k > t.i - 4: k -= 1
            if not (k >= 1 and doc[k].lower_ in ("a", "an") and doc[k - 1].lower_ == "as"): return None
            anchor = doc[k - 1]
            if nxt is None or nxt.lower_ != "of": nxt = t           # 'as a result, Y': the phrase ends at the noun
        elif item == "on account of":
            if t.i == 0 or doc[t.i - 1].lower_ != "on": return None
            anchor = doc[t.i - 1]
        elif item == "in light of":     # 'in light of' and 'in the light of' (decided 2 Nov 2026)
            if t.i >= 1 and doc[t.i - 1].lower_ == "in": anchor = doc[t.i - 1]
            elif t.i >= 2 and doc[t.i - 1].lower_ == "the" and doc[t.i - 2].lower_ == "in":
                anchor = doc[t.i - 2]
                po = prep_object(nxt) if nxt.dep_ == "prep" else None
                if po is not None and po[0] == "noun" and _literal_light(po[1]):
                    return rec(item, "literal light", group="other sense")    # 'in the light of the moon'
            else: return None
        else:
            anchor = t
        _nb = lambda r: (r.__setitem__("_not_before", new_item and _not_before(doc, anchor.i)) or r)
        h = anchor.head
        steps = 0
        while h.i in range(anchor.i, nxt.i + 1) and h.head is not h and steps < 4: h = h.head; steps += 1
        if h.pos_ in ("NOUN", "PROPN", "PRON") and anchor.dep_ in ("amod", "prep", "acl"):
            return _nb(rec(item, "effect on the left (noun)", h))
        if lem(h) == "be" and predicate(h) is not h and predicate(h).i not in range(anchor.i, nxt.i + 1):
            return _nb(clause_effect(item, h, "effect on the left (clause)"))
        if lem(h) == "be" and anchor.dep_ in ("acomp", "attr", "prep"):
            # 'the delay was not due to rain': a negative form of the connective (1 Nov 2026), not a negated effect
            r = rec(item, "effect on the left (noun)", subject_of(h), None,
                    "negated" if not config.NEG_RULES and is_neg(h) else "main"); r["polarity"] = polarity(word_neg(h))
            return _nb(r)
        if h.pos_ in ("VERB", "AUX", "ADJ"):
            return _nb(clause_effect(item, h, "effect on the left (clause)"))
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
        # 'X is not associated with Y': a negative form, kept (CAUSESENT_NEG_RULES, 1 Nov 2026); before: left out
        grp = "negated" if not config.NEG_RULES and (is_neg(t) or (t.dep_ == "acomp" and is_neg(t.head))) else "main"
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
        w = {"sufficient for": "sufficient", "necessary for": "necessary", "required for": "required", "enough for": "enough",
             "responsible for": "responsible"}[item]
        if t.lower_ != w: return None
        cands = kids + [c for c in t.head.children if c.i > t.i and t.head is not t]
        p = next((c for c in cands if c.dep_ == "prep" and c.lower_ == "for"), None)
        vfor = next((c for c in cands if c.pos_ in ("VERB", "AUX") and any(g.lower_ == "for" and g.dep_ in ("mark", "prep") for g in c.children)), None)
        if vfor is not None:   # 'sufficient for the plant to grow'
            N = next((g for g in vfor.children if g.dep_ in SUBJ), None)
            return rec(item, "ADJ + for N to V", N, vfor, "negated" if eff_neg(vfor) else "main")
        if p is None: return None
        po = prep_object(p)
        if po is None: return None
        if po[0] == "noun":
            inner = next((g for g in po[1].children if g.dep_ in ("acl", "relcl") and any(x.lower_ == "to" for x in g.children)), None)
            if inner is not None: return rec(item, "ADJ + for N to V", po[1], inner, "negated" if eff_neg(inner) else "main")
            return rec(item, "ADJ + for N", po[1])
        return rec(item, "ADJ + for V-ing", po[1], po[2], "negated" if eff_neg(po[2]) else "main")
    return None

# ---- trigger finding
_BY_LEMMA = {}
for _it, (_lem, _kind) in ITEMS.items():
    _BY_LEMMA.setdefault(_lem, []).append(_it)
_WORD_TRIG = {"because": ["because of", "because (clause)"], "due": ["due to"], "thanks": ["thanks to"], "owing": ["owing to"],
              "sufficient": ["sufficient for"], "necessary": ["necessary for"], "required": ["required for"], "enough": ["enough for"],
              "result": ["as a result (of)", "result of"], "results": ["result of"],
              "consequence": ["as a consequence (of)", "consequence of"], "consequences": ["consequence of"],
              "so": ["so"], "therefore": ["therefore"], "thus": ["thus"], "hence": ["hence"], "consequently": ["consequently"],
              "since": ["since"], "as": ["as", "in order to"], "for": ["for"], "given": ["given that"],
              "account": ["on account of"], "light": ["in light of"], "reason": ["reason"], "reasons": ["reason"],
              "why": ["why"], "cause": ["the cause of"], "causes": ["the cause of"], "effect": ["the effect of X on Y"],
              "effects": ["the effect of X on Y"], "responsible": ["responsible for"], "order": ["in order to"],
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
