"""Colloquial 'cause = because (adopted 30 Oct 2026; switch CAUSESENT_CAUSE_FILTER, on by default).

Only the bare form 'cause' (also 'cause, cause, CAUSE) can mean 'because'; causes / caused / causing are always the verb.
Corpus taggers and spaCy often tag 'cause = because as a verb ('I left cause I was tired'), and the clause after it is
then parsed as the verb's complement. The verb 'cause' takes N, N + to V and N + N, but never a finite clause, a bare
infinitive, an adjective or a participle, so those lines land in four frames the verb does not license:

    CAUSE_BECAUSE_FRAMES = V + that-clause, V + N + bare V, V + N + ADJ, V + N + V-ed, and (1 Nov 2026) V + N

A bare-form 'cause' line is moved out of the results (group 'because') when it is in one of these frames AND the
word-order rule below does not call it a verb (frames.py). Frames alone remove many real verbs in edited prose, where
spaCy also misparses them into these frames ('could cause radioactive contamination over a much wider area than ...');
the word-order rule alone over-flags in edited prose ('and cause it to collapse'). Hand checks of the combination
(30 Oct 2026; 'because' or noun lines removed / real verbs removed): AskReddit 47/1 of 159 scored lines, COCA 10/0,
BNC 19/4, enTenTen21 60/7. 'Because' lines parsed as V + N ('cause ppls reactions') stay in.

classify(doc, i): the word-order rule (tags only, no parse) -> 'verb' | 'because' | 'other' (the game title Just
Cause; the noun: 'cause of death', 'have cause to', 'cause and effect'); None if doc[i] is not a bare 'cause'.
On 100 blind-labelled AskReddit lines: precision 0.92, recall 0.92 for 'because' (alone, without the frame condition).
Development and test samples: partly_cause/spacy/code/because_v2/ in the project folder.

Order of the checks:
  1. other (title / noun)
  2. verb after a modal, 'to', gonna/wanna, 'd/'ll                     'can cause', 'would really cause', 'to cause'
  3. verb after a relative pronoun after a noun                         'things that cause those feelings are ...'
  4. because if a clause follows (subject + finite verb) or a wh-word / sentence adverb follows
                                                                        'cause her parents were ...', 'cause then I ...'
  5. verb after do-support, a plural or 1st/2nd-person subject          'it doesn't cause', 'they cause', 'drugs cause'
  6. verb after a singular subject or and/or, if only a noun phrase follows   'nugget on earth cause various vet bills'
  7. verb in a fragment: 'cause' at the start or after a comma + a noun phrase ('Cause tailgaters' engines to die')
  8. otherwise because
"""
SKIP = {"also", "really", "actually", "definitely", "certainly", "probably", "possibly", "potentially", "likely",
        "only", "even", "still", "never", "not", "n't", "always", "usually", "often", "sometimes", "eventually",
        "ultimately", "inevitably", "easily", "directly", "indirectly", "further", "simply", "both", "surely",
        "typically", "generally", "frequently", "rarely", "commonly", "undoubtedly", "invariably", "necessarily",
        "all", "each"}
DO = {"do", "does", "did", "don't", "doesn't", "didn't", "dont", "doesnt", "didnt"}
MODALISH = {"to", "'d", "'ll", "ll", "d", "gonna", "wanna", "gotta", "would", "could", "can", "will", "may", "might",
            "should", "must", "shall", "cannot", "can't", "won't", "wouldn't", "couldn't", "shouldn't", "cant", "wont"}
REL = {"that", "which", "who", "whom"}
SUBJ_PL = {"i", "you", "we", "they", "u", "ya", "these", "those"}
SUBJ_3SG = {"he", "she", "it", "this", "that"}
WH = {"why", "if", "when", "how", "what", "where", "who", "whenever", "now", "then", "again", "apparently", "otherwise",
      "honestly", "seriously", "obviously", "clearly", "maybe", "frankly", "basically", "literally", "like", "idk",
      "fuck", "once", "unlike", "as", "of", "y", "u", "ur", "im", "ive", "id", "theres", "thats", "its", "it's"}
FIN = {"VBD", "VBZ", "VBP", "MD"}
BE_HAVE = {"be", "is", "was", "are", "were", "been", "being", "'s", "have", "has", "had", "give", "gives", "find", "show", "no", "good", "just", "probable", "reasonable"}
STOP = {"if", "because", "since", "when", "while", "although", "though", "unless", "whether", "until", "that", "which",
        "who", "whom", "where", "so", "but"}
TEMPORAL = {"day", "days", "time", "times", "night", "week", "year", "morning", "today", "tonight", "yesterday", "tomorrow", "someday"}
FRAG_NP = {"a", "an", "another", "no", "more", "some", "any", "all", "serious", "severe", "major", "mass", "massive",
           "significant", "great", "huge", "permanent", "damage", "problems", "harm", "issues", "chaos", "trouble"}
SUBJ_START = {"PRP", "EX", "PRP$", "DT", "NN", "NNS", "NNP", "NNPS", "CD", "JJ", "JJR", "JJS", "PDT", "WP"}

def _norm(t):
    return t.lower_.strip("'’`")

def _clause_follows(doc, i, span=6):
    """Does a clause start right after doc[i]: a wh-word / sentence adverb, or a subject followed by a finite verb?"""
    if i + 1 >= len(doc): return False
    n = doc[i + 1]
    if n.lower_ in WH: return True
    if n.lower_ in TEMPORAL or (i + 2 < len(doc) and doc[i + 2].lower_ in TEMPORAL and n.tag_ in ("CD", "DT", "JJ")): return True
    if n.tag_ not in SUBJ_START: return False
    for k in range(i + 1, min(len(doc), i + 1 + span)):
        t = doc[k]
        if t.is_punct and t.text not in ("'", "’"): return False
        if k > i + 1 and (t.tag_ in ("WDT", "WRB", "TO") or t.lower_ in STOP): return False
        if t.tag_ in FIN: return True
    return False

def classify(doc, i):
    t = doc[i]
    if _norm(t) != "cause": return None
    nxt = doc[i + 1] if i + 1 < len(doc) else None
    nx2 = doc[i + 2] if i + 2 < len(doc) else None
    prev_raw = doc[i - 1] if i >= 1 else None
    j = i - 1                                             # left neighbour, skipping adverbs and not/n't
    while j >= 0 and (doc[j].lower_ in SKIP or (doc[j].tag_ in ("RB", "RBR", "RBS") and doc[j].lower_ not in ("just", "so"))):
        j -= 1
    prev = doc[j] if j >= 0 else None
    pl = prev.lower_ if prev is not None else ""
    clause = _clause_follows(doc, i)
    # 1. neither the verb nor 'because'
    if prev_raw is not None and prev_raw.lower_ == "just" and t.text[:1].isupper() and (nxt is None or nxt.like_num or nxt.is_punct):
        return "other"
    if nxt is not None and nxt.lower_ in ("of", "for") and prev is not None and pl in BE_HAVE:
        return "other"                                    # 'that would be cause for concern'
    if nxt is not None and nxt.lower_ in ("of", "for") and not clause:
        if prev is not None and (prev.tag_ in ("DT", "PRP$", "JJ", "JJS", "JJR") or pl in ("the", "a", "main", "root")):
            return "other"
    if nxt is not None and nxt.lower_ == "and" and nx2 is not None and nx2.lower_ in ("effect", "effects"):
        return "other"
    if prev is not None and prev.tag_ in ("DT", "PRP$") and pl not in ("that", "this", "these", "those", "all", "no", "any", "some") and not clause:
        return "other"                                    # 'a cause', 'the cause', 'their cause'
    if prev is not None and pl in ("have", "has", "had", "no", "good") and nxt is not None and nxt.lower_ == "to":
        return "other"
    # 2. modal / to
    if prev is not None and (prev.tag_ == "MD" or pl in MODALISH):
        return "verb"
    # 3. relative pronoun after a noun: 'things that cause', 'bacteria which cause'
    if prev is not None and pl in REL and j >= 1 and doc[j - 1].tag_.startswith("NN"):
        return "verb"
    # 4. a clause follows
    if clause:
        return "because"
    # 5. do-support, plural or 1st/2nd-person subject, relative pronoun
    if prev is not None and (pl in DO or pl in SUBJ_PL or pl in REL or prev.tag_ in ("NNS", "NNPS", "WDT", "WP")):
        return "verb"
    # 6. singular subject or and/or, followed only by a noun phrase
    if prev is not None and nxt is not None and (pl in SUBJ_3SG or prev.tag_ in ("NN", "NNP") or pl in ("and", "or")):
        if nxt.tag_ in ("DT", "JJ", "JJR", "JJS", "NN", "NNS", "CD", "PDT") and nxt.lower_ not in WH:
            return "verb"
    # 7. a fragment starting with 'cause' + an effect-like noun phrase: 'cause another financial crisis', 'cause no other problems'
    start = prev_raw is None or prev_raw.is_punct or (prev_raw.lower_ == "just" and (i < 2 or doc[i - 2].is_punct))
    k = i + 1
    while k < len(doc) and doc[k].tag_ in ("RB", "RBR", "RBS"): k += 1
    if start and k < len(doc) and (doc[k].lower_ in FRAG_NP or doc[k].tag_ in ("DT", "JJ", "JJR", "JJS", "NN", "NNS", "PDT", "CD")):
        return "verb"
    # 8.
    return "because"


# V + N added 1 Nov 2026 (Dean), for all corpora ('cause ain't nobody got time ...); switch CAUSESENT_CAUSE_FILTER_VN.
# Test (1 Nov 2026): it newly removes BNC 45 lines (about 2 really because / noun), enTenTen21 58 (about 10), COCA 5 (2),
# AskReddit 35 of 1,500 random bare-form lines (7): about four real verbs lost per 'because' line caught.
from . import config
CAUSE_BECAUSE_FRAMES = {"V + that-clause", "V + N + bare V", "V + N + ADJ", "V + N + V-ed"} | ({"V + N"} if config.CAUSE_FILTER_VN else set())


def is_because_line(t, frame):
    """True if token t (a trigger of the item 'cause') is colloquial 'cause = because by the combined rule."""
    if frame not in CAUSE_BECAUSE_FRAMES or _norm(t) != "cause": return False
    return classify(t.doc, t.i) in ("because", "other")
