"""Rules for the causal uses of so, since, as, for, therefore, consequently, thus, hence (written and tested 1 Nov 2026 in
sense_check_20261101/: 350 AskReddit lines labelled before the rules were written; used in the pipeline from 2 Nov 2026).
Changes for the pipeline (2 Nov 2026): 'so as to' is never 'so' (it belongs to 'in order to'); 'as' and 'for' count only
in their frames, as their queries are narrowed in every corpus: 'for' after a comma ('..., for he was tired'), 'as'
after a comma or at the start of a sentence ('..., as it was late'; 'As it was late, ...'). 'out of' was dropped. causal_sense(doc, i, j) -> (label, reason): label 'causal' or
'not causal'; i..j = the token span of the word. Precision first: unclear cases are 'not causal'. spaCy en_core_web_sm."""
from nltk.corpus import wordnet as wn
FINITE = {"VBD", "VBZ", "VBP", "MD"}
SUBJ = ("nsubj", "nsubjpass", "expl", "csubj")
def _clause_follows(doc, k, window=4):
    """A finite clause starts at token k: a subject within the next tokens followed by a finite verb, or a finite verb."""
    for t in doc[k:min(len(doc), k + window)]:
        if t.tag_ in FINITE or (t.pos_ in ("VERB", "AUX") and t.tag_ == "VBD"): return True
        if t.is_punct: return False
    return False
def _verb_before(doc, k):
    return any(t.pos_ in ("VERB", "AUX") for t in doc[:k] if t.sent == doc[k].sent)
def _sent_initial(doc, k):
    s = doc[k].sent.start
    return all(t.is_punct or t.lower_ in ("and", "but", "well", "oh", "ok", "okay", "yeah", "lol") for t in doc[s:k])
def _next(doc, k, n=1): return doc[k + n] if k + n < len(doc) else None
def _prev(doc, k, n=1): return doc[k - n] if k - n >= 0 else None

# ---- so (incl. 'so that')
SO_FIXED_PREV = {"think", "thought", "hope", "hoped", "guess", "say", "said", "do", "did", "does", "doing", "done", "or",
                 "even", "if", "not", "believe", "suppose", "seems", "seem", "appears", "imagine", "just", "ever", "why"}
SO_FIXED_NEXT = {"far", "much", "many", "long", "on", "what", "be", "called", "forth", "sorry", "help"}
def _so(doc, k):
    t = doc[k]; nx = _next(doc, k); pv = _prev(doc, k)
    if t.text == "SO" or (nx is not None and pv is not None and pv.lower_ in ("my", "your", "his", "her", "our", "their")):
        return "not causal", "SO = significant other"
    if nx is not None and nx.lower_ == "as" and _next(doc, k, 2) is not None and _next(doc, k, 2).lower_ == "to":
        return "not causal", "so as to (counts for 'in order to')"
    if nx is not None and nx.lower_ == "that" and (nx.dep_ == "mark" or nx.pos_ == "SCONJ"):
        return "causal", "so that"
    if nx is not None and (nx.pos_ in ("ADJ", "ADV") or nx.tag_ in ("JJ", "JJR", "RB")) and nx.lower_ not in ("i", "then"):
        return "not causal", "intensifier (so + adjective/adverb)"
    if t.head.pos_ in ("ADJ", "ADV") and t.dep_ == "advmod" and t.head.i > k: return "not causal", "intensifier"
    if nx is not None and nx.lower_ in SO_FIXED_NEXT: return "not causal", f"fixed: so {nx.lower_}"
    if pv is not None and pv.lower_ in SO_FIXED_PREV: return "not causal", f"fixed: {pv.lower_} so"
    if nx is None or nx.is_punct: return "not causal", "so at the end / alone"
    if _sent_initial(doc, k): return "not causal", "sentence-initial So (discourse marker)"
    if _verb_before(doc, k) and _clause_follows(doc, k + 1): return "causal", "clause, so, clause"
    return "not causal", "no clause on both sides"

# ---- for: only the clause-linking conjunction ('..., for he was tired')
SUBJ_PRON = {"i", "you", "he", "she", "we", "they", "it", "there", "this", "that", "these", "those"}
def _for(doc, k):
    nx = _next(doc, k); pv = _prev(doc, k)
    if nx is None: return "not causal", "end"
    if pv is None or pv.text != ",": return "not causal", "not after a comma (frame ', for')"
    if nx.lower_ in ("example", "instance", "one", "now", "sure", "ever", "once", "starters", "all", "real", "good", "free",
                     "more", "less", "years", "decades", "centuries", "months", "weeks", "days", "hours", "a", "an", "the"):
        return "not causal", f"fixed: for {nx.lower_}"     # 'for example', 'for the first time' ... (added 2 Nov 2026)
    if not _verb_before(doc, k): return "not causal", "no clause before"
    if nx.lower_ in SUBJ_PRON and _next(doc, k, 2) is not None and _next(doc, k, 2).tag_ in FINITE:
        return "causal", "for + subject pronoun + finite verb"
    # PRP$ added 2 Nov 2026 ('..., for her son was dead')
    if (nx.pos_ in ("DET", "NOUN", "PROPN") or nx.tag_ == "PRP$") and any(x.tag_ in FINITE and x.dep_ in ("ROOT", "conj", "ccomp", "advcl")
                                                     for x in doc[k + 1:min(len(doc), k + 5)]) \
            and any(x.dep_ in SUBJ for x in doc[k + 1:min(len(doc), k + 4)]):
        return "causal", "for + noun subject + finite verb"
    return "not causal", "preposition 'for'"

# ---- out of: a motive ('out of curiosity'), WordNet feeling / motive nouns or a short list
MOTIVE_EXTRA = {"curiosity", "boredom", "habit", "necessity", "spite", "respect", "pity", "desperation", "frustration",
                "kindness", "jealousy", "guilt", "concern", "gratitude", "obligation", "ignorance", "laziness", "compassion",
                "pride", "malice", "revenge", "love", "fear", "anger", "courtesy", "convenience", "principle", "loyalty",
                "sympathy", "interest", "nostalgia", "instinct", "reflex", "self-defense", "self-defence", "duty", "spite",
                "generosity", "goodwill", "desire", "need", "panic", "shame", "embarrassment", "impatience", "sadness"}
def _out_of(doc, k):
    of = doc[k + 1] if k + 1 < len(doc) else None
    obj = next((c for c in of.children if c.dep_ == "pobj"), None) if of is not None else None
    if obj is None:
        obj = next((x for x in doc[k + 2:min(len(doc), k + 5)] if x.pos_ in ("NOUN", "PROPN")), None)
    if obj is None: return "not causal", "no object noun"
    w, l = obj.lower_, obj.lemma_.lower()
    if w in MOTIVE_EXTRA or l in MOTIVE_EXTRA: return "causal", f"motive: {l}"
    s = wn.synsets(l, pos="n")[:1]
    if s and s[0].lexname() in ("noun.feeling", "noun.motive"): return "causal", f"motive ({s[0].lexname()}): {l}"
    return "not causal", f"not a motive: {l}"

# ---- therefore, consequently, thus, hence: causal, except fixed non-causal phrases
TIME_NOUNS = {"year", "years", "month", "months", "week", "weeks", "day", "days", "hour", "hours", "century", "centuries"}
def _adverb(doc, k, w):
    nx = _next(doc, k); pv = _prev(doc, k)
    if w == "thus" and nx is not None and nx.lower_ == "far": return "not causal", "thus far"
    if w == "hence" and pv is not None and pv.lower_ in TIME_NOUNS: return "not causal", "temporal: N hence"
    return "causal", w

# ---- since: temporal unless shown otherwise? No: causal unless a temporal cue (precision for 'causal' relies on the cues)
MONTHS_DAYS = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
               "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "monday",
               "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "christmas", "easter", "spring", "summer",
               "autumn", "fall", "winter"}
PERFECT_AUX = {"have", "has", "had", "'ve", "'s", "'d", "’ve", "’s", "’d"}
YOUTH = {"kid", "child", "baby", "teenager", "teen", "little", "young", "toddler", "boy", "girl", "born"}
def _main_of(t):
    sub = t.head
    return sub.head if sub.dep_ in ("advcl", "ccomp", "conj", "prep", "npadvmod") and sub.head is not sub else None
def _perfect(v):
    if v is None: return False
    if any(c.dep_ in ("aux", "auxpass") and c.lower_ in PERFECT_AUX for c in v.children): return True
    if v.lower_ == "been" or (v.tag_ == "VBN" and any(c.lower_ in PERFECT_AUX for c in v.children)): return True
    return False
SINCE_TIME_NOUNS = {"year", "month", "week", "day", "decade", "hour", "minute", "second", "while", "time", "age", "century",
                    "period", "season", "night", "morning", "evening", "generation"}
def _past(v):
    """The since-clause is in the past: a past-tense verb, or an auxiliary 'was / were / had' ('was diagnosed')."""
    return v.tag_ == "VBD" or any(c.dep_ in ("aux", "auxpass") and c.lower_ in ("was", "were", "had") for c in v.children)
def _since(doc, k):
    t = doc[k]; nx = _next(doc, k); pv = _prev(doc, k)
    if pv is not None and pv.lower_ in ("ever", "long"): return "not causal", f"{pv.lower_} since"
    if nx is None or nx.is_punct: return "not causal", "since at the end (adverb: 'since then')"
    if nx.lower_ in ("then", "now", "birth", "childhood", "forever", "ages", "yesterday", "last", "before", "after"):
        return "not causal", f"since {nx.lower_}"
    if nx.like_num or nx.tag_ == "CD" or nx.ent_type_ in ("DATE", "TIME") or nx.lower_ in MONTHS_DAYS:
        return "not causal", "since + date/number"     # month and day names added 2 Nov 2026 (the pipeline runs spaCy without NER)
    # three temporal cues added 2 Nov 2026 (COCA spot check: 'which since has closed', 'It's been years since I made',
    # 'the most important decision since we decided'); tested on COCA, enTenTen21 and AskReddit lines (fix_20261102/)
    if (pv is not None and pv.lower_ in ("has", "have", "had", "'ve", "’ve", "been")) or nx.lower_ in ("has", "have", "had", "been"):
        return "not causal", "perfect auxiliary next to since ('has since closed', 'which since has closed')"
    if not _clause_follows(doc, k + 1, window=5):
        return "not causal", "since + noun phrase / gerund (preposition)"
    if (t.head.dep_ in ("advcl", "relcl", "acl") and t.head.head.pos_ == "NOUN"
            and t.head.head.lemma_.lower() in SINCE_TIME_NOUNS):
        return "not causal", "time noun + since + clause ('years since I made', 'a while since')"
    if (pv is not None and pv.pos_ == "NOUN" and _past(t.head)
            and any(x.tag_ in ("JJS", "RBS") or (x.lower_ in ("first", "last") and x.tag_ == "JJ")
                    for x in pv.subtree if x.i < pv.i)):
        return "not causal", "superlative noun + since + past clause ('the most important decision since we decided')"
    main = _main_of(t)
    if _perfect(main): return "not causal", "perfect tense in the main clause"
    if main is not None and main.lemma_ == "be" and any(c.lower_ in ("been",) for c in main.children):
        return "not causal", "it's been ... since"
    clause = t.head
    if any(x.lower_ in YOUTH for x in clause.subtree) and any(x.lower_ in ("was", "were") for x in clause.subtree):
        return "not causal", "since I was a kid / young"
    if any(x.lower_ == "ago" for x in clause.subtree): return "not causal", "since ... ago"
    return "causal", "since + clause, no temporal cue"

# ---- as
AS_FIXED_NEXT = {"if", "though", "well", "long", "soon", "far", "much", "many", "for", "to", "of", "per", "such", "usual",
                 "always", "opposed", "in", "a", "an", "the", "part", "is", "are", "was", "were"}
AS_FIXED_PREV = {"such", "same", "so", "just", "even", "known", "regarded", "seen", "used", "serve", "serves", "served",
                 "work", "works", "worked", "act", "acts", "acted", "referred", "describe", "described", "treat", "treated"}
COMMENT_VERBS = {"say", "said", "mention", "mentioned", "know", "knew", "see", "saw", "note", "noted", "state", "stated",
                 "point", "pointed", "expect", "expected", "predict", "predicted", "suggest", "suggested", "imagine",
                 "guess", "think", "thought", "recall", "remember", "write", "wrote", "put", "read", "hear", "heard",
                 "understand", "suspect", "told", "tell", "explained", "can", "could", "should", "would", "do", "did",
                 # added 2 Nov 2026 (Sketch Engine check): 'as I mistakenly believed', 'as some might claim'
                 "believe", "believed", "claim", "claimed", "argue", "argued", "assume", "assumed", "report", "reported",
                 "show", "showed", "shown", "describe", "described", "discuss", "discussed", "indicate", "indicated",
                 "hope", "hoped", "fear", "feared", "intend", "intended", "explain", "seem", "seems", "seemed", "appear",
                 "appears", "appeared", "imply", "implied", "propose", "proposed", "recommend", "recommended"}
STATIVE = {"be", "have", "know", "seem", "own", "need", "want", "like", "understand", "believe", "live", "lack",
           "belong", "contain", "depend", "include", "require", "cost", "matter", "mean", "prefer", "hate", "love"}
def _sent_start(doc, k):
    t = doc[k]
    return t.i == t.sent.start or all(x.is_punct for x in doc[t.sent.start:k])
def _as(doc, k):
    t = doc[k]; nx = _next(doc, k); pv = _prev(doc, k)
    if not ((pv is not None and pv.text == ",") or _sent_start(doc, k)):
        return "not causal", "not after a comma or at the start of a sentence (frames ', as' / 'As')"
    if nx is None or nx.is_punct: return "not causal", "as at the end"
    if nx.lower_ in AS_FIXED_NEXT: return "not causal", f"fixed: as {nx.lower_}"
    if pv is not None and pv.lower_ in AS_FIXED_PREV: return "not causal", f"fixed: {pv.lower_} as"
    if nx.pos_ in ("ADJ", "ADV", "NUM", "DET") or nx.tag_ in ("JJ", "RB", "CD"): return "not causal", "as + adjective/adverb/number (comparison)"
    # added 2 Nov 2026 (AskReddit check): role 'as' before a noun phrase ('As candy flavoring they are fine', 'As someone who')
    n2 = _next(doc, k, 2)
    if nx.lower_ in ("someone", "somebody", "anyone", "anybody", "one", "everyone", "everybody"):
        return "not causal", "as + someone (role)"
    if nx.lower_ == "luck": return "not causal", "as luck would have it"
    if nx.pos_ in ("NOUN", "PROPN") and n2 is not None and (n2.pos_ in ("NOUN", "PROPN", "ADP", "PRON") or n2.lower_ in ("who", ",")):
        return "not causal", "as + noun phrase (role)"
    if t.pos_ == "ADP" or t.dep_ == "prep": return "not causal", "as + noun phrase (role / comparison)"
    if any(x.lower_ == "as" for x in doc[max(0, k - 4):k]): return "not causal", "as ... as comparison"
    if not _clause_follows(doc, k + 1, window=4): return "not causal", "no finite clause after as"
    clause = t.head
    # added 2 Nov 2026 (Sketch Engine check): comment clauses without a subject ('as can be seen', 'as will become
    # apparent', 'as mentioned'); correlative 'As X, so Y'; elliptical comparisons ('as it is to you', 'as there is with all
    # of those': be / do / have with no complement but a prepositional phrase or adverb)
    if any(c.dep_ == "acomp" and c.lower_ in ("sure", "aware") for c in clause.children):
        return "not causal", "comment clause (as I'm sure you know, as you are aware)"
    if not any(c.dep_ in ("nsubj", "nsubjpass", "expl", "csubj") for c in clause.children):
        return "not causal", "comment clause without a subject (as can be seen)"
    if any(x.lower_ == "so" and x.i > clause.i and x.i - 1 >= 0 and doc[x.i - 1].text == "," for x in doc[k:min(len(doc), k + 30)]):
        return "not causal", "correlative: As X, so Y"
    if clause.lemma_ in ("be", "do", "have") and not any(c.dep_ in ("dobj", "attr", "acomp", "xcomp", "ccomp", "neg", "oprd") for c in clause.children):
        return "not causal", "elliptical comparison (as it is to you)"
    if clause.lower_ in COMMENT_VERBS or clause.lemma_ in COMMENT_VERBS and not any(c.dep_ in ("dobj", "ccomp", "xcomp", "attr", "acomp") for c in clause.children):
        return "not causal", "comment clause (as I said / as you know)"
    if clause.lemma_ == "be" and not any(c.dep_ in ("attr", "acomp", "prep", "advmod") for c in clause.children):
        return "not causal", "manner: as it is"
    if any(c.tag_ == "VBG" for c in [clause]) or any(c.dep_ == "aux" and c.lemma_ == "be" for c in clause.children):
        return "not causal", "progressive (temporal)"
    if clause.lemma_ in STATIVE or any(c.dep_ == "aux" and c.tag_ == "MD" for c in clause.children):
        if any(c.dep_ in ("dobj", "attr", "acomp", "xcomp", "ccomp", "prep", "neg", "advmod") for c in clause.children):
            return "causal", "as + stative/modal clause with a complement"
    return "not causal", "as + other clause (temporal or manner)"

def causal_sense(doc, i, j):
    w = " ".join(t.lower_ for t in doc[i:j])
    if w == "so": return _so(doc, i)
    if w == "for": return _for(doc, i)
    if w == "out of": return _out_of(doc, i)
    if w == "since": return _since(doc, i)
    if w == "as": return _as(doc, i)
    if w in ("therefore", "consequently", "thus", "hence"): return _adverb(doc, i, w)
    raise ValueError(w)
