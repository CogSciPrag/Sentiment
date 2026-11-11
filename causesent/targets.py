"""The 101 studied expressions (80 analysed first + 21 added 2 Nov 2026) and the word forms used to pre-select sentences.

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
 "owing to": ("owing", "conn"),
 # unified 1 Nov 2026: 'as a result of N' and 'as a result, CLAUSE' are one item (formerly 'as a result of')
 "as a result (of)": ("as a result (of)", "conn"),
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
 # ---- the 21 items added on 2 Nov 2026 (word_list_v6.tsv; queries in docs/queries.tsv). For items that are not verbs
 # the first field is a label, not a lemma, so that no baseline verb is excluded by accident (e.g. 'give', 'light').
 # result connectives: effect = the clause they introduce (kind 'adv')
 "so": ("so", "adv"), "therefore": ("therefore", "adv"), "thus": ("thus", "adv"), "hence": ("hence", "adv"),
 "consequently": ("consequently", "adv"),
 "as a consequence (of)": ("as a consequence (of)", "conn"),
 # reason connectives: effect = the main clause
 "since": ("since", "sub"), "as": ("as", "sub"), "for": ("for", "sub"), "given that": ("given that", "sub"),
 "on account of": ("on account of", "conn"), "in light of": ("in light of", "conn"),
 # explanation
 "explain": ("explain", "verb"), "reason": ("reason (noun)", "noun"), "why": ("why", "why"),
 "the cause of": ("the cause of", "noun"), "the effect of X on Y": ("the effect of", "noun"),
 # effect on the left / attribution
 "result of": ("result of", "noun"), "consequence of": ("consequence of", "noun"),
 "responsible for": ("responsible", "adj"),
 # purpose
 "in order to": ("in order to", "purpose"),
 # added 3 Nov 2026 (add-on run, determine_plan_20261103.md): causal sense of 'determine' only (frames.determine_other_sense)
 "determine": ("determine", "verb"),
}
# Words that are tested items but stay IN the baselines (Dean, 3 Nov 2026: 'determine' was already in the v9 baselines; the
# baselines are left as they are). The baseline scripts (steps 4, 7, 8) leave BASELINE_KEEP out of their exclusion sets.
BASELINE_KEEP = {"determine"}
# Display labels of the figures where they differ from the item name (3 Nov 2026, Dean): the item "the cause of" covers every
# "cause of" / "causes of" + noun (also "a cause of", "causes of", "cause of death"), so it is shown as "cause(s) of".
DISPLAY = {"the cause of": "cause(s) of"}
SYMMETRIC = {"associated with", "correlated with"}
# Connectives (decided 1 Nov 2026): compared with Glass's (2023) baseline of all other subordinating conjunctions
# (spaCy SCONJ; generic item 'other conjunctions', scored on the main clause like 'because (clause)') as the main
# comparison, and with the verb baseline as a second one. The planned connectives (since, as, so, therefore, ...) join
# this set when they are added.
CONNECTIVES = {it for it, (lem, kind) in ITEMS.items() if kind in ("conn", "adv", "sub", "purpose")}
# Noun items (decided 1 Nov 2026): also compared with a NOUN baseline, all other nouns in 'the N of X' and
# 'X is the N of Y' (X scored) in the baseline samples (steps 4, 7, 8), reported alongside the verb baseline.
NOUN_ITEMS = {it for it, (lem, kind) in ITEMS.items() if kind == "noun"}
# nouns left out of that baseline: the tested nouns and the quantity / part words that are looked through (frames.py)
NOUN_BASE_EXCLUDE = {"cause", "effect", "result", "consequence", "reason", "explanation", "outcome", "upshot"}
# SCONJ words left out of that baseline (decided 1 Nov 2026): every word we test, as Glass (2023) leaves 'because' out
# of its own baseline ('sconj != "because"'), plus the colloquial forms of 'because' ('cause, cuz, cos, which Glass's code
# kept). Tested conjunction words: because, since, as, for, so, why, and 'that' in 'so that' / 'given that' (frames.py).
SCONJ_EXCLUDE = {"because", "cause", "'cause", "’cause", "cuz", "cos", "coz", "bc", "b/c", "becuz", "bcuz", "bcoz",
                 "since", "as", "for", "so", "why"}
SCONJ_EXCLUDE_AFTER = {"that": {"so", "given", "order"}}    # 'that' only when part of a tested expression ('so that', 'given that', 'in order that')
PREVENTION = {"prevent", "stop", "block", "hinder", "inhibit", "save", "protect", "restrain", "deter", "discourage", "fail to"}
EFFECT_LEFT = {"result from", "stem from", "arise from", "depend on", "because of", "due to", "thanks to", "owing to",
               "as a result (of)", "because (clause)", "as a consequence (of)", "since", "as", "for", "given that",
               "on account of", "in light of", "result of", "consequence of"}
# Samples that are downloaded but NOT used (step 6 skips them by sample name):
#  - the old 'as a result of' samples, replaced by the 'as a result (of)' resample of 2 Nov 2026 (double counting)
#  - COCA '. as' (decided 2 Nov 2026: 'as' is left out of COCA, whose ', as' cannot be sampled: > 500,000 hits)
SKIP_SAMPLES = {"se_ententen21__as_a_result_of", "se_bnc__as_a_result_of", "se_coca__as_a_result_of", "se_coca__as__q2"}
# Words left out of the AskReddit pre-selection unless in their frames: 'as' and 'for' only after a comma, and 'as' at
# the start of a sentence (decided 2 Nov 2026: their queries are narrowed to the frames in every corpus)

# Word groups used to summarise the results (provisional grouping)
WORD_GROUPS = {
 "1 Cause-type causatives": ["cause", "make", "force", "compel", "get", "induce", "render", "bring about", "give rise to",
                             "trigger", "spark", "provoke", "prompt"],
 "2 Leads-to / results-in": ["lead to", "result in", "contribute to", "produce", "generate", "create", "yield"],
 "3 Ensuring (sufficiency for an outcome)": ["ensure", "guarantee", "secure", "make sure", "determine"],
 "4 Sufficiency adjectives": ["sufficient for", "enough for"],
 "5 Necessity / requirement": ["require", "necessitate", "demand", "necessary for", "required for", "depend on"],
 "6 Enabling / permitting / helping": ["enable", "allow", "permit", "let", "help", "assist", "facilitate", "foster", "promote",
                                        "make a difference to"],
 "7 Persuasion / motivation": ["encourage", "convince", "inspire", "motivate", "persuade", "oblige", "stimulate", "tempt", "intend"],
 "8 Prevention †": ["prevent", "stop", "block", "hinder", "inhibit", "discourage", "save", "protect", "restrain", "deter"],
 "9 Effect on the left / connectives ←": ["result from", "stem from", "arise from", "because of", "due to", "thanks to",
                                           "owing to", "as a result (of)", "because (clause)", "result of", "consequence of"],
 "10 Attribution / influence": ["blame", "credit", "account for", "affect", "influence", "responsible for"],
 "11 Implicatives": ["manage to", "fail to", "succeed in"],
 "12 Correlation (non-causal comparison) ≈": ["associated with", "correlated with"],
 # added 2 Nov 2026 (groups provisional, as in word_list_v6.tsv)
 "13 Result connectives": ["so", "therefore", "thus", "hence", "consequently", "as a consequence (of)"],
 "14 Reason connectives ←": ["since", "as", "for", "given that", "on account of", "in light of"],
 "15 Explanation": ["explain", "reason", "why", "the cause of", "the effect of X on Y"],
 "16 Purpose": ["in order to"],
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
FORMS += ["explain", "explains", "explained", "explaining"]
# the items added on 2 Nov 2026 (words, not verb forms); 'as' and 'for' only in their frames (see above)
NEW_WORDS = ["so", "therefore", "thus", "hence", "consequently", "consequence", "consequences", "since", "given",
             "account", "light", "reason", "reasons", "why", "cause", "causes", "effect", "effects", "result", "results",
             "responsible", "order"]
RX = re.compile(r"\b(" + "|".join(sorted(set(FORMS) | set(NEW_WORDS))) + r")\b", re.I)
RX_AS_FOR = re.compile(r",\s*(as|for)\b|^\W*as\b", re.I)
def has_target(sent): return RX.search(sent) is not None or RX_AS_FOR.search(sent) is not None
