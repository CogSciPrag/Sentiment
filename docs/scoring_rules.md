# How the effect is found and scored

This document describes the rules implemented in `causesent/frames.py` (finding the effect) and
`causesent/scoring.py` (scoring it and deciding which lines count). Only the sentiment of the **effect** is analysed;
the cause side is not scored.

## 1. Finding the expression

Each line is parsed with spaCy 3.6.0 (`en_core_web_sm` 3.6.0). For concordance samples (enTenTen21, BNC, COCA), the
sentence containing the search hit is rebuilt from the left context, the hit and the right context, and only the
searched expression at the hit is analysed. For Reddit, every studied expression in each selected sentence is
analysed. Each occurrence gives one record, tagged with a **frame** (the construction it occurs in).

## 2. The effect after the expression

Most expressions (*cause*, *ensure*, *lead to*, *prevent* …) introduce the effect after them.

| Construction | Frame | Scored | Example |
|---|---|---|---|
| object, no complement | `V + N` | head noun N | *cause an accident* → *accident* |
| two objects | `V + N + N (double object)` (and the parses below) | the second noun only | *caused me a lot of stress* → *stress*; *made the kitchen a mess* → *mess* |
| object + to-infinitive | `V + N + to V` | N and V | *force the victims to flee* → *victim*, *flee* |
| object + bare infinitive (*make*, *let*, *help* only) | `V + N + bare V` | N and V | *let the kids play* → *kid*, *play* |
| object + adjective | `V + N + ADJ` | N and the adjective | *made him happy* → *he*, *happy* |
| object + participle | `V + N + V-ed` | N and the participle | *got the car fixed* → *car*, *fix* |
| object + from/for/in + V-ing | `V + N + from V-ing` etc. | N and the gerund | *save the child from drowning* → *child*, *drown* |
| that-clause | `V + that-clause` | the clause's subject and its verb, or the predicate after *be* (*be going to V* → V, 2 Nov 2026) | *ensure that everyone is safe* → *everyone*, *safe* |
| to-infinitive, no object | `V + to V` | V, or the predicate after *be* (2 Nov 2026; also in `V + N + to V`) | *manage to escape* → *escape*; *make sure to be consistent* → *consistent* |
| preposition + noun | `V + PREP + N` | head noun | *lead to war* → *war* |
| preposition + gerund clause | `V + PREP + V-ing` | the gerund's subject and the gerund (or the gerund alone) | *resulted in prices rising* → *price*, *rise* |
| adjective + for | `ADJ + for N` / `ADJ + for N to V` / `ADJ + for V-ing` | what follows *for* | *sufficient for survival* → *survival* |

**Two objects.** With two objects the second is scored, for every verb (no exemptions; decided 29 October 2026): it
is the thing caused, while the first is who it happens to (*caused my family a lot of stress* → *stress*, *saved you
the trouble* → *trouble*, *made him president* → *president*, *rendered the plan a failure* → *failure*). spaCy parses
this construction in four ways, all covered: indirect + direct object (`V + N + N (double object)`); a noun complement
with its own subject (`…, clause parse`: *caused them problems*); two direct objects (`…, two-object parse`: *made the
kitchen a mess*); and, for every verb except those that take a bare infinitive (*make*, *let*, *help*, and among the
baseline verbs perception verbs such as *see*, *hear*, *watch*), the second noun misread as a bare verb (`…, bare-V parse`: *causing people pain*, with *pain* tagged as a
verb). If the second word is unrated, the line gets no score. Degree and duration phrases (*a lot*, *the most*, *the
whole show*) and appositions (*you guys*, *us all*) are not second objects.

Expression-specific rules:

- ***blame* and *credit***: the noun after *with/for* is the effect (*blamed the driver for the crash* → *crash*). For
  every other verb the direct object stays the effect (*caused concern for the companies* → *concern*).
- ***make a difference to***: the object of *to* (*made a difference to people's lives* → *life*). Only with *a*
  (optionally an adjective: *a big difference*); not *make all the difference*, not *make a difference in*. An adjective
  on *difference* itself is not scored (*a huge difference*), as no expression's own adjectives are (decided 1 Nov 2026,
  also for the planned *the effect of*, *result of*, *consequence of*).
- ***make sure***: the that-clause, or V alone in *make sure to V*.
- ***give rise to***, ***bring about***: the object of *to* / *about*.

## 3. The effect before the expression

*result from*, *stem from*, *arise from*, *depend on* (effect = the subject), and *because of*, *due to*, *thanks to*,
*owing to*, *as a result of*, *because* + clause (effect = what the phrase attaches to):

- effect is a noun: its head noun (*the delay stemmed from a strike* → *delay*; *delays due to snow* → *delay*);
- effect is a clause: its subject N and its main verb or predicate V (*the game was cancelled because of rain* →
  *game*, *cancel*; *the roads were icy due to the storm* → *road*, *icy*).

## 4. Correlation expressions

*associated with* and *correlated with* are a non-causal comparison class. There is no effect, so both sides are
scored: X in the N slot and Y in the V slot (*smoking is associated with cancer* → *smoking*, *cancer*;
*the risks associated with smoking* → *risk*, *smoking*). Negated uses (*is not associated with*) are negative forms
(section 6); the active *associate X with Y* (a mental association) and the social *associate with someone* are kept
out of the results.

## 5. Scoring

- Every scored word is looked up as a single word in the Hedonometer (1 = sad, 9 = happy, 5 = neutral), **form first,
  then lemma** (decided 1 Nov 2026; switch `CAUSESENT_FORM_FIRST`): the word as it occurs (lowercased) if the
  Hedonometer lists it (*problems* 2.92, *lives* 6.22, *species* 5.76, *means* 5.64), otherwise its spaCy lemma
  (en_core_web_sm 3.6.0); a word with neither is skipped, and a line with no rated word is not scored. This follows
  the Hedonometer's own entries and avoids lemmatiser errors (*species* -> *specie*, *means* -> *mean* 3.68,
  *taxes* -> *taxis*). The first results, and Glass's (2023) direct-object scores, used the lemma only; Glass scored
  her clause measures by word form. For the Warriner variant the same rule uses the Warriner list; for the neutral-band
  variant the form/lemma choice is made in the full Hedonometer, then neutral words are dropped.
  The list is **labMT-en-v2** (hedonometer.org, 2020-03-28), the list Glass (2023) used: all 5,772 of her word scores
  match it. It has 10,187 words, with the same scores as labMT 1.0 (Dodds et al. 2011) for every shared word; it adds
  17 pandemic words (*covid*, *lockdown*, *quarantine* …) and drops 52 Twitter hashtags and handles. The first results
  used `labMT2english.txt` (10,222 words, the labMT 1.0 list). Words not in the lexicon are skipped.
- **Pronouns** are not scored in the N slot (rule switch 4 below); a line whose N is a pronoun rests on V.
- **Quantity words** are looked through to the noun after *of* (rule switch 3 below): *a lot of questions* →
  *question*.
- **Coordination** is averaged: *cause fires and floods* → mean of *fire* and *flood*; *forced them to pack and leave*
  → V = mean of *pack* and *leave*.
- **Line score**: with a complement, the mean of N and V (or whichever of the two is rated); without one, N.
- Three **measures** are reported: N only, V only, and the line score (NV).
- An expression's mean pools all its lines (frequent frames count more), and is reported only with **at least 30
  scored lines** in a corpus.

Four **scoring variants** (`causesent/scoring.py`):

| Variant | Lexicon and rule |
|---|---|
| `raw` (main) | Hedonometer, single words, no exclusion |
| `band` | Hedonometer, words scoring 4–6 dropped (the neutral band of Dodds et al. 2011, also used by Glass) |
| `adj` | Hedonometer; the noun's adjectives and noun compounds are averaged into N (*poor performance* → *poor*, *performance*) |
| `war` | Warriner, Kuperman & Brysbaert (2013) valence norms instead of the Hedonometer |

## 6. Which lines count

- **Main group only.** Passives (*the fire was caused by lightning*) and negated effects form separate groups and
  are reported separately. A negated effect is negation *inside* the effect: *persuaded him not to go*, *ensure that
  nobody gets hurt*, *make sure nothing breaks*, *the game was not cancelled because of rain* (not / n't / never on the
  effect's verb, or a negative subject).
- ***into*-phrases** (decided 1 Nov 2026): for the verbs of inducing (*force, compel, persuade, convince, tempt, provoke,
  induce, encourage, inspire, motivate, stimulate, oblige*) the *into*-phrase attached to the verb is the effect: *forced
  him into resigning* → *resign*, *provoked them into a war* → *war*; not places or things moved into (*forced them into
  his truck*), not *force one's way into*.
- **Positive and negative forms (decided 1 Nov 2026).** Negation of the studied word itself is kept (*didn't persuade
  him to go*, *does not require*), as in Glass (2023), and every line is tagged with its polarity. Negative form = not /
  n't / never on the word (*didn't cause*, *is not sufficient for*), or a negative subject (*nothing causes*, *no one
  forced me*, *none of these changes led to*, *neither option ensures*), or *no* / *zero* on the effect noun (*caused no
  damage*, *led to no good*); for the connectives, *X was not due to /
  because of Y* (a negated full verb, *was not cancelled because of rain*, stays a negated effect). Results are given
  three times: positive forms, negative forms, and both combined, each against the same split of the baseline
  (`polarity` column of `pvalues_all.tsv`). Combined is the headline and what the figures show; sample sizes refer to
  the combination. Before 1 Nov 2026 the negated effect-left verbs (*did not result from*), *X was not due to Y* and
  *is not associated with* were left out of the results; they are now negative forms. These changes, and negative
  subjects in negated effects, are the switch `CAUSESENT_NEG_RULES` (the polarity column is recorded either way).
- ***make* and *get***: only their causative frames count (*make* N + adjective / bare verb / participle; *get* N +
  adjective / participle / to-infinitive). *make bread* or *get a job* are left out of the results, and so are all
  their lines with two objects (*made him a sandwich*, *made him president*, *got me a job*; version B, the headline).
  A follow-up (version C, `pipeline/11_make_get_followup.py`) adds back *make* N + N unless the second noun is
  something made for someone (`CREATED_OBJECTS` in `causesent/scoring.py`: *sandwich, dinner, coffee, card* …), and
  *get* N + N only when the subject of *get* is clearly not a person (*having a degree won't get you a job*, *an extra
  skill that gets me raises*; not *a friend got me a job*, not imperatives, proper names or unknown subjects). The two
  word lists were built from 160,000 AskReddit *make* / *get* lines.
- **Colloquial *'cause* = *because*** (rule switch 5 below) is left out of the results of *cause* (group `because`). Extending it to the V + N
  frame (switch `CAUSESENT_CAUSE_FILTER_VN`) was tested on 1 Nov 2026 and left OFF: it removed about four real verbs per
  *because* line caught (results_v7_coca_test/sense_filters_20261101/).
- **Non-causal senses** (switch `CAUSESENT_SENSE_FILTERS`, decided 1 Nov 2026; group `other sense`, left out): *render*
  only with a result (*rendered it useless*, *rendered the plan a failure*; not *render a service / a verdict / an image*);
  *secure* not fastening or guarding (*secure X to / with / from Y*, or a thing, place, body part, animal or place name as
  object; *secure funding* counts; added 2 Nov 2026: computer-security objects WordNet does not class as things, *secure
  your server / network / data / account / information / connection ...*); *save* only rescue and prevention (*save X from Y*, or a person, group, animal, life,
  the world … as object; not *save the file*, *save money / time*, *save you time*); *contribute to* not giving or taking
  part (an organisation, publication, discussion or fund as object).
- **Removal effects** are tagged (column `removal`), not excluded: the effect is something going away (*make the pain
  go away*, *helped the war end*). Word list (confirmed 1 Nov 2026): *go away, disappear, stop, vanish, fade, end, pass*.
- **Prevention words** (*prevent, stop, block, hinder, inhibit, discourage, save, protect, restrain, deter*, and
  *fail to*) are pooled with the others but labelled: their effect is what does *not* happen.
- **No frame restrictions**: every use of every expression counts; frames are used for tagging and for the
  frame-by-frame tables, not for filtering (apart from *make* and *get*).

## 7. Baselines and statistics

Each corpus has its own baseline: all other lexical verbs, analysed with the same verb rules as a generic item
'other verbs' (be/have/do and the trigger lemmas of the studied expressions excluded).

| Corpus | Baseline sample |
|---|---|
| enTenTen21, BNC | 10,000 random concordance lines of `[tag="VV.*"]` per corpus |
| AskReddit | a random 3-million-word subsample of the first 10 million words of January 2018 |
| COCA | every lexical verb in 5,500 random lines around 11 frequent function words (COCA offers no random sample of all verbs) |

**Connective baseline (decided 1 Nov 2026).** The connectives (*because (clause)*, *because of*, *due to*, *thanks to*,
*owing to*, *as a result of*, and the planned ones when added) are also compared with Glass's (2023) baseline: every
other subordinating conjunction (spaCy `SCONJ`: *when, if, while, after, before, until, since, as, so, though, that,
whether* …), scored on its main clause as the generic item 'other conjunctions', the same way *because (clause)* is
scored. This is their main comparison; the verb baseline is reported as a second one (column `baseline_type` in
`pvalues_all.tsv`). Left out of it, as Glass leaves *because* out of its own baseline: every conjunction word we test (*because, since,
as, for, so, why*, and *that* in *so that* / *given that*), and the colloquial forms of *because* (*'cause, cuz, cos* …),
which Glass's code kept. The conjunctions come from the same samples as the verb baseline: every SCONJ in the AskReddit sample, in the
sentence around each enTenTen21 / BNC baseline line, and in the COCA anchor lines (where the anchors *after*, *when*,
*where* are skipped in their own samples).

Each expression is compared with its corpus's baseline by Welch's t-test (95% CI of the difference, Cohen's d with
the pooled standard deviation), with Holm correction within each scoring × corpus × measure × polarity × baseline.

## 8. Rule switches (October 2026)

Five refinements were agreed in October 2026 after the first results were produced. They are **on by default**. Each
can be switched off with its environment variable set to `0` (see `causesent/config.py`); `CAUSESENT_RULES=main`
switches all five off, and with `CAUSESENT_LABMT=labMT2english.txt` the code reproduces the first results.

1. **Double objects** (`CAUSESENT_DOUBLE_OBJECT`). In *caused me any significant concerns*, *saved you the trouble*,
   *guarantee them passage*, the second noun is scored (*concern*, *trouble*, *passage*) instead of the first. Three
   of spaCy's parses of this construction are covered (indirect + direct object; a noun complement with its own
   subject, as in *caused them problems*; two direct objects, as in *made the kitchen a mess*). Applied to every verb,
   *make*, *get*, *render* and *let* included (until 29 October 2026 these four were exempt); *make* and *get* N + N
   lines are then left out of the headline results (section 6). The two-direct-object parse also turns up for
   hyphenated words split by the tokenizer (*re - election*), appositions, lists and duration phrases, so it counts only
   when the two nouns are adjacent and the second is a common noun opening with a determiner, quantifier or adjective
   (*guaranteeing same-sex couples the right to marry*, *rendering Scott a quadriplegic*); on COCA it then applies to
   25 lines, about two-thirds of them true double objects. Not applied when the second noun is a degree or
   duration phrase (*a lot*, *the most*, *the whole show*) or an apposition (*you guys*, *us all*).
2. **Bare-verb parse of a double object** (`CAUSESENT_BARE_V_DOUBLE`; added 30 October 2026). *causing people pain*,
   *cause students anguish*: spaCy often tags the second noun as a bare verb with the first as its subject
   (`V + N + bare V`). Except after verbs that take a bare infinitive (*make*, *let*, *help*; among the baseline verbs,
   perception verbs: *watched the kids play*), this is read as a double object and only the second word is scored
   (*pain* 2.10, not the mean of *people* and *pain*); if it is unrated, the line gets no score. It applies only when
   the 'verb' looks like a noun: no auxiliary, no dependents of its own besides its subject and noun modifiers, and
   directly after its subject (so *ensure they would be able* and *demand the government do anything* are untouched).
   `BARE_INF_VERBS` in `causesent/frames.py`.
3. **Quantity words** (`CAUSESENT_QUANTITY`). A head noun that names an amount, a part or a kind is replaced by the
   noun of its *of*-phrase: *a lot of questions* → *question*, *72 percent of convictions* → *conviction*, *part of its
   popularity* → *popularity*, *a kind of organization* → *organization*. Money words count as amounts (added 1 Nov
   2026): *3 million dollars of damage* → *damage*. Pure amounts with no *of*-phrase (*22 percent*,
   *much more*, numbers) are not scored; part and kind words (*part, type, form, level, degree, number, rest* …) are
   looked through only when *of* follows (so *metal parts* and *a music education degree* stay as they are). The word
   lists are `QUANT_DROP` and `QUANT_OF` in `causesent/frames.py`.
4. **No pronouns** (`CAUSESENT_NO_PRONOUNS`). Pronouns are not scored in the N slot, subjects and objects alike
   (*it stemmed from*, *ensure that you …*, *caused me to …*), including demonstratives (*this, that*), relatives
   (*which, who, what*) and indefinites (*someone, nothing, everyone* …). The line then rests on V; with no V it gets
   no score. The list is `PRONOUNS` in `causesent/frames.py`.

5. **Colloquial *'cause* = *because*** (`CAUSESENT_CAUSE_FILTER`; added 30 October 2026). Corpus taggers and spaCy
   often tag *'cause* = *because* as a verb (*I only eat here 'cause it's free*); our *cause* samples, drawn with
   verb-tagged queries, contain it in every corpus. The verb *cause* takes N, N + to V and N + N, but never a finite
   clause, a bare infinitive, an adjective or a participle, and when *'cause* = *because* is parsed as the verb, the
   clause after it lands in exactly those frames. A bare-form *cause* line (*cause*, *'cause*; never *causes*,
   *caused*, *causing*) is left out (group `because`) when it is in `V + that-clause`, `V + N + bare V`,
   `V + N + ADJ` or `V + N + V-ed` **and** a word-order rule (`causesent/because_filter.py`) does not call it a verb:
   the verb is required after a modal, *to*, *do*, a plural or first/second-person subject, a relative pronoun after
   a noun, or a singular subject followed only by a noun phrase; it is *because* when a clause, a wh-word or a
   sentence adverb follows. Neither condition alone is enough: in edited prose spaCy also puts real verbs into these
   frames (*could cause radioactive contamination over a much wider area than …*), and the word-order rule over-flags
   (*and cause it to collapse*). Hand checks of the combination ('because' or noun lines removed / real verbs removed):
   AskReddit 47 / 1 of 159 scored lines, COCA 10 / 0, BNC 19 / 4, enTenTen21 60 / 7. *Because* lines parsed as
   `V + N` (*cause ppls reactions*) stay in.

On COCA, switches 1, 3 and 4 as first tested (20 October 2026, before *make*, *get*, *render* and *let* lost their
exemption) changed no expression's mean by more than 0.34 (raw, NV; the largest shifts are *depend on* +0.33 and *blame* −0.27): *cause* moves from 4.52 to 4.34
(Cohen's d against the baseline from −1.14 to −1.20, still the most negative of the 80), and *ensure*, *guarantee*,
*secure* and *make sure* stay above the baseline. The rank order of the 80 expressions is almost unchanged (Spearman
0.98). Apply the switches to the baselines as well (run every step with the same settings).

## 9. The 21 expressions added on 2 November 2026

Word list: `word_list_v6.tsv` (101 items = 80 + 21). Queries, hits and samples: `docs/queries.tsv` (rows of 2 Nov 2026).
Code: `causesent/frames.py` (section "The 21 items added on 2 Nov 2026") and `causesent/senses.py` (sense rules).
Every line still gets a frame, a group and a polarity, as in sections 1-8.

**Kinds and effects**

| expressions | effect scored | example |
|---|---|---|
| so, therefore, thus, hence, consequently | the clause they introduce (subject N + predicate V); 'so that Y' counts for *so*; 'thus V-ing' → the participle and its object; 'hence N' → the noun | *It was icy, so **the roads closed***; *hence **the name*** |
| since, as, for, given that | the main clause | *Since the store was closed, **we went home*** |
| as a result (of), as a consequence (of), on account of, in light of | the clause or noun they attach to (as for *because of*) | *As a result, **thousands died*** |
| explain | as any verb: object or clause | *The drought explains why **the crops failed*** |
| why | the clause after *why* (statements and questions) | *That's why **the project failed*** |
| reason | the relative clause (*the reason (why/that) Y*) or the for-object (*the reason for N*) | *the reason **he left*** |
| the cause of | the noun (or V-ing clause) after *of* | *the cause of **cancer*** |
| result of, consequence of, the effect of | only a named effect: the subject in *X is the N of Y*, or a noun predicate in *the N of Y is X*, or the clause in *the N of Y is that X* | ***Damage** was the effect of the storm* |
| responsible for | as *sufficient for*: the for-object | *responsible for **the deaths*** |
| in order to / that / for, so as to | the purpose verb and its object (or the clause after *in order that*; N + V in *in order for N to V*) | *in order to **win the prize*** |

Not counted: adjective predicates of result / consequence / effect (*the effect was significant*: they say how the effect
was, not what it was), appositions (*X, the result of Y*: mostly lists and misparses), *reason to V*, *for some reason*,
*the reason is that X* (X is the cause).

**Senses** (`senses.py`, tested 1 Nov 2026 in `sense_check_20261101/`): non-causal *so* (degree, *so much*, *I think so*,
sentence-initial *So*), temporal *since* (dates, perfect main clause, *ever since*), non-causal *as* (comparison, role,
*as I said*, progressive / temporal clauses), prepositional *for* (incl. *for example*), *thus far*, *N hence* → group
'other sense'. *as* and *for* count only in their frames, as their queries are narrowed in every corpus: *, as* or
sentence-initial *As*; *, for*. Additions on 2 Nov 2026: *for* + possessive subject (*..., for her son was dead*); month and
day names after *since* (the pipeline runs without NER); temporal *since* next to a perfect auxiliary (*has since
closed*, *which since has closed*), after a time noun (*years since I made*, *a while since*, *the first time since*) and
after a superlative noun with a past-tense clause (*the most important decision since we decided*; tests in
`fix_20261102/`); *for example / instance / the ...* excluded; *given* as a verb
(*evidence was given that*, *a given*) excluded; literal *in the light of the moon / fire / screen* excluded.

**Counted once** (group 'overlap'): *as a result of* lines found in a *result of* sample count for *as a result (of)*;
*as a consequence of* likewise; *the reason why* counts for *reason*, not *why*; *so as to* counts for *in order to*, not *so*.

**Negative forms** (approved by Dean, 2 Nov 2026; `negative_forms_draft_20261102.md`): the claim is denied — *not* before
the connective (*not on account of X*, *not in order to*), a negated copula (*X is not the cause of Y*, *that's not why Y*,
*the delay was not on account of*), *no* on the noun (*there is no reason why Y*), a negated or negative-subject verb or
adjective (*nothing explains Y*, *nobody is responsible for Y*). *so / therefore / thus / hence / consequently* have none.
Negation inside the effect stays group 'negated'. *no* / *zero* on the effect noun is a negative form, as before.

**Baselines**: connectives (*so* ... *in order to*) are compared with the conjunction baseline (main) and the verb
baseline; the noun items (*the cause of, the effect of, result of, consequence of, reason*) with a NOUN baseline — every
other noun in *the N of X* (X scored) or *X is the N of Y* (X scored) in the baseline samples (generic item 'other nouns';
tested nouns and quantity / part words left out) — and the verb baseline. *explain* is now left out of the verb baseline
(tested word). *that* after *in order* is left out of the conjunction baseline.

**COCA**: its search syntax has no optional slot, so items with one were searched with several queries (*in light of* /
*in the light of*; *as a result* / *as a|an _j* result*; ...), and the samples are combined in proportion to each query's
hits (`scoring.QUERY_FREQ`; overlap lines dropped first). *as* is left out of COCA (decided 2 Nov 2026: *, as* has more
than 500,000 hits, which COCA will not display). The old *as a result of* samples are not used (`targets.SKIP_SAMPLES`).

**Checks (2 Nov 2026)**: the 80 earlier expressions give identical COCA results with the new code (n and means of all
comparisons); the COCA verb baseline loses 11 *explain* lines (5.6058 → 5.6061). Spot checks of scored lines (8-10 per
expression, COCA and AskReddit, Claude's judgement): see query_checklist.txt.


## 10. 'determine' (causal sense only; add-on item of 3 Nov 2026)
Item `determine` (word group 3), run as an add-on to the 101 expressions; 'determine' stays in the baselines (targets.BASELINE_KEEP).
Frames: those of every verb (object noun, whether / if / wh-clauses, to V, ...); passives are excluded as for every verb.
Left out as 'other sense' (frames.determine_other_sense): (1) 'determine that [clause]' with an explicit *that* (epistemic in about 99% of
the lines); (2) an infinitive ('to determine', 'in order to determine'); (3) an ability modal (*can / could determine*: about 93% epistemic);
(4) 'determined to V' (resolved); (5) no subject of its own (imperative, gerund after a preposition), or a participle / imperative
mis-parsed as a modifier of a noun such as *time*, *way*, *step*; (6) a person, organisation, agent pronoun (*I, you, we, they, who, those*)
as subject (an agent decides); (7) an inquiry or instrument noun as subject (*test, study, analysis, scan, algorithm, ...*:
epistemic); (8) a verb conjoined with an epistemic verb (*look at X and determine whether*). A relative clause or participle takes the
noun it modifies as its subject (*factors that determine*). Precision (own reading of random scored lines): AskReddit about 90% of 60,
BNC 42 of 45, enTenTen21 about 76% of 45 before rules (1) and (4) were added, COCA 31 of 37; recall: about 2 causal lines in 40 left-out lines.
