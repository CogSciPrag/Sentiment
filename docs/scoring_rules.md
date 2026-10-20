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
| object + to-infinitive | `V + N + to V` | N and V | *force the victims to flee* → *victim*, *flee* |
| object + bare infinitive | `V + N + bare V` | N and V | *let the kids play* → *kid*, *play* |
| object + adjective | `V + N + ADJ` | N and the adjective | *made him happy* → *he*, *happy* |
| object + participle | `V + N + V-ed` | N and the participle | *got the car fixed* → *car*, *fix* |
| object + from/for/in + V-ing | `V + N + from V-ing` etc. | N and the gerund | *save the child from drowning* → *child*, *drown* |
| that-clause | `V + that-clause` | the clause's subject and its verb, or the predicate after *be* | *ensure that everyone is safe* → *everyone*, *safe* |
| to-infinitive, no object | `V + to V` | V | *manage to escape* → *escape* |
| preposition + noun | `V + PREP + N` | head noun | *lead to war* → *war* |
| preposition + gerund clause | `V + PREP + V-ing` | the gerund's subject and the gerund (or the gerund alone) | *resulted in prices rising* → *price*, *rise* |
| adjective + for | `ADJ + for N` / `ADJ + for N to V` / `ADJ + for V-ing` | what follows *for* | *sufficient for survival* → *survival* |

Expression-specific rules:

- ***blame* and *credit***: the noun after *with/for* is the effect (*blamed the driver for the crash* → *crash*). For
  every other verb the direct object stays the effect (*caused concern for the companies* → *concern*).
- ***make a difference to***: the object of *to* (*made a difference to people's lives* → *life*). Only with *a*
  (optionally an adjective: *a big difference*); not *make all the difference*, not *make a difference in*.
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
*the risks associated with smoking* → *risk*, *smoking*). Negated uses, the active *associate X with Y* (a mental
association) and the social *associate with someone* are kept out of the results.

## 5. Scoring

- Every scored word is a **lemma**, looked up as a single word in the Hedonometer (labMT 2.0; 1 = sad, 9 = happy,
  5 = neutral). Words not in the lexicon are skipped. Pronouns and demonstratives are scored as they are, as in
  Glass (2023) (but see rule switch 3 below).
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
  nobody gets hurt*, *the game was not cancelled because of rain*.
- **Negation of the causal word itself is kept** (*didn't persuade him to go*, *does not require*), as in Glass (2023).
- ***make* and *get***: only their causative frames count (*make* N + adjective / bare verb / participle; *get* N +
  adjective / participle / to-infinitive). *make bread* or *get a job* are left out of the results.
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

Each expression is compared with its corpus's baseline by Welch's t-test (95% CI of the difference, Cohen's d with
the pooled standard deviation), with Holm correction within each scoring × corpus × measure.

## 8. Rule switches (October 2026)

Three refinements were tested after the main results were produced. They are **off by default**, so the code
reproduces the main results. Switch them on with environment variables (see `causesent/config.py`):

1. **Double objects** (`CAUSESENT_DOUBLE_OBJECT=1`). In *caused me any significant concerns*, *saved you the trouble*,
   *guarantee them passage*, the second noun is scored (*concern*, *trouble*, *passage*) instead of the first. Both of
   spaCy's parses of this construction are covered (indirect + direct object, and a noun complement with its own
   subject, as in *caused them problems*). Not applied to *make*, *get*, *render* (*made him president*, *render it a
   loser* are resultatives) or *let*, nor when the second noun is a degree or duration phrase (*a lot*, *the most*,
   *the whole show*) or an apposition (*you guys*, *us all*).
2. **Quantity words** (`CAUSESENT_QUANTITY=1`). A head noun that names an amount, a part or a kind is replaced by the
   noun of its *of*-phrase: *a lot of questions* → *question*, *72 percent of convictions* → *conviction*, *part of its
   popularity* → *popularity*, *a kind of organization* → *organization*. Pure amounts with no *of*-phrase (*22 percent*,
   *much more*, numbers) are not scored; part and kind words (*part, type, form, level, degree, number, rest* …) are
   looked through only when *of* follows (so *metal parts* and *a music education degree* stay as they are). The word
   lists are `QUANT_DROP` and `QUANT_OF` in `causesent/frames.py`.
3. **No pronouns** (`CAUSESENT_NO_PRONOUNS=1`). Pronouns are not scored in the N slot, subjects and objects alike
   (*it stemmed from*, *ensure that you …*, *caused me to …*), including demonstratives (*this, that*), relatives
   (*which, who, what*) and indefinites (*someone, nothing, everyone* …). The line then rests on V; with no V it gets
   no score. The list is `PRONOUNS` in `causesent/frames.py`.

On COCA, the three together change no expression's mean by more than 0.34 (raw, NV; the largest shifts are *depend on* +0.33 and *blame* −0.27): *cause* moves from 4.52 to 4.34
(Cohen's d against the baseline from −1.14 to −1.20, still the most negative of the 80), and *ensure*, *guarantee*,
*secure* and *make sure* stay above the baseline. The rank order of the 80 expressions is almost unchanged (Spearman
0.98). Apply the switches to the baselines as well (run every step with the same settings).
