"""The 80 studied expressions (78 causal, 2 correlational) and the word forms used to pre-select sentences.

ITEMS        expression -> (trigger lemma, kind): the lemma the parser must find, and how the effect is located
             (see frames.py). Kinds: 'verb' (all uses), 'phrasal' (verb + particle/preposition), 'conn' (connective),
             'adj' (adjective + for), 'impl' (implicative + to/in), 'assoc' (correlation: both sides scored)
PREVENTION   expressions whose effect is what does NOT happen (pooled with the others, but labelled)
EFFECT_LEFT  expressions whose effect precedes them ('the delay stemmed from a strike')
SYMMETRIC    correlation expressions, where X and Y are both scored
has_target() quick regular-expression filter used to keep only candidate sentences from Reddit
"""
import re

# item -> (trigger lemma, kind). kind: 'verb' (all uses), 'phrasal' (verb + particle/preposition),
# 'conn' (connective), 'adj' (adjective + for), 'impl' (implicative + to/in)
ITEMS = {
 # causing / sufficiency
 "cause": ("cause", "verb"), "make": ("make", "verb"), "make sure": ("make", "phrasal"), "get": ("get", "verb"),
 "let": ("let", "verb"), "force": ("force", "verb"), "compel": ("compel", "verb"), "ensure": ("ensure", "verb"),
 "guarantee": ("guarantee", "verb"), "secure": ("secure", "verb"), "trigger": ("trigger", "verb"), "spark": ("spark", "verb"),
 "provoke": ("provoke", "verb"), "prompt": ("prompt", "verb"), "induce": ("induce", "verb"), "render": ("render", "verb"),
 "bring about": ("bring", "phrasal"), "give rise to": ("give", "phrasal"), "produce": ("produce", "verb"),
 "generate": ("generate", "verb"), "create": ("create", "verb"),
 # enabling / influencing
 "enable": ("enable", "verb"), "allow": ("allow", "verb"), "permit": ("permit", "verb"), "help": ("help", "verb"),
 "facilitate": ("facilitate", "verb"), "foster": ("foster", "verb"), "promote": ("promote", "verb"),
 "contribute to": ("contribute", "phrasal"), "influence": ("influence", "verb"), "affect": ("affect", "verb"),
 # preventing
 "prevent": ("prevent", "verb"), "stop": ("stop", "verb"), "block": ("block", "verb"), "hinder": ("hinder", "verb"),
 "inhibit": ("inhibit", "verb"),
 # causal link
 "lead to": ("lead", "phrasal"), "result in": ("result", "phrasal"), "result from": ("result", "phrasal"),
 "stem from": ("stem", "phrasal"), "arise from": ("arise", "phrasal"), "depend on": ("depend", "phrasal"),
 "require": ("require", "verb"), "necessitate": ("necessitate", "verb"),
 # connectives
 "because of": ("because", "conn"), "due to": ("due", "conn"), "thanks to": ("thanks", "conn"),
 "owing to": ("owing", "conn"), "as a result of": ("result", "conn"),
 # persuasion / motivation
 "encourage": ("encourage", "verb"), "convince": ("convince", "verb"), "inspire": ("inspire", "verb"),
 "motivate": ("motivate", "verb"), "persuade": ("persuade", "verb"), "oblige": ("oblige", "verb"),
 "stimulate": ("stimulate", "verb"), "tempt": ("tempt", "verb"), "intend": ("intend", "verb"),
 "discourage": ("discourage", "verb"),
 # helping / influencing
 "assist": ("assist", "verb"), "make a difference to": ("make", "phrasal"),
 # preventing
 "save": ("save", "verb"), "protect": ("protect", "verb"), "restrain": ("restrain", "verb"), "deter": ("deter", "verb"),
 # attribution / producing
 "blame": ("blame", "verb"), "credit": ("credit", "verb"), "account for": ("account", "phrasal"), "yield": ("yield", "verb"),
 # requirement
 "demand": ("demand", "verb"),
 # connective / adjective
 "because (clause)": ("because", "conn"), "sufficient for": ("sufficient", "adj"), "necessary for": ("necessary", "adj"),
 "required for": ("require", "adj"), "enough for": ("enough", "adj"),
 # implicative
 "manage to": ("manage", "impl"), "fail to": ("fail", "impl"), "succeed in": ("succeed", "impl"),
 # correlation, a non-causal comparison class. Symmetric: X and Y are both scored.
 "associated with": ("associate", "assoc"), "correlated with": ("correlate", "assoc"),
}
SYMMETRIC = {"associated with", "correlated with"}
PREVENTION = {"prevent", "stop", "block", "hinder", "inhibit", "save", "protect", "restrain", "deter", "discourage", "fail to"}
EFFECT_LEFT = {"result from", "stem from", "arise from", "depend on", "because of", "due to", "thanks to", "owing to",
               "as a result of", "because (clause)"}

# Word groups used to summarise the results (provisional grouping)
WORD_GROUPS = {
 "1 Cause-type causatives": ["cause", "make", "force", "compel", "get", "induce", "render", "bring about", "give rise to",
                             "trigger", "spark", "provoke", "prompt"],
 "2 Leads-to / results-in": ["lead to", "result in", "contribute to", "produce", "generate", "create", "yield"],
 "3 Ensuring (sufficiency for an outcome)": ["ensure", "guarantee", "secure", "make sure"],
 "4 Sufficiency adjectives": ["sufficient for", "enough for"],
 "5 Necessity / requirement": ["require", "necessitate", "demand", "necessary for", "required for", "depend on"],
 "6 Enabling / permitting / helping": ["enable", "allow", "permit", "let", "help", "assist", "facilitate", "foster", "promote",
                                        "make a difference to"],
 "7 Persuasion / motivation": ["encourage", "convince", "inspire", "motivate", "persuade", "oblige", "stimulate", "tempt", "intend"],
 "8 Prevention †": ["prevent", "stop", "block", "hinder", "inhibit", "discourage", "save", "protect", "restrain", "deter"],
 "9 Effect on the left / connectives ←": ["result from", "stem from", "arise from", "because of", "due to", "thanks to",
                                           "owing to", "as a result of", "because (clause)"],
 "10 Attribution / influence": ["blame", "credit", "account for", "affect", "influence"],
 "11 Implicatives": ["manage to", "fail to", "succeed in"],
 "12 Correlation (non-causal comparison) ≈": ["associated with", "correlated with"],
}

IRREG = {"arise": ["arises", "arose", "arisen", "arising"], "bring": ["brings", "brought", "bringing"],
  "get": ["gets", "got", "gotten", "getting"], "give": ["gives", "gave", "given", "giving"], "lead": ["leads", "led", "leading"],
  "let": ["lets", "letting"], "make": ["makes", "made", "making"], "stem": ["stems", "stemmed", "stemming"],
  "stop": ["stops", "stopped", "stopping"], "permit": ["permits", "permitted", "permitting"],
  "compel": ["compels", "compelled", "compelling"], "deter": ["deters", "deterred", "deterring"]}
def forms(l):
    if l in IRREG: return [l] + IRREG[l]
    if l.endswith("e"): return [l, l + "s", l + "d", l[:-1] + "ing"]
    return [l, l + "s", l + "ed", l + "ing"]
_verbs = {lem for lem, kind in ITEMS.values() if kind in ("verb", "phrasal", "impl")} | {"require"}
FORMS = sorted({f for l in _verbs for f in forms(l)} | {"because", "due", "thanks", "owing", "sufficient", "necessary", "enough"}
               | {f for l in ("associate", "correlate") for f in forms(l)})
RX = re.compile(r"\b(" + "|".join(FORMS) + r")\b", re.I)
def has_target(sent): return RX.search(sent) is not None
