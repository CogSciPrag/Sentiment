# sentiment

This project investigates the sentiment profile of causal words; in particular, why *cause* prefers
negative-sentiment effects.

Code for a corpus study of the sentiment of the effects that causal expressions introduce, by Dean McHugh (University of Edinburgh).

Glass (2023) found that the verb *cause* favours negative-sentiment complements (*cause problems*, *cause damage*).
This study tests one implication across 78 causal expressions and 2 correlational ones: if *cause*'s negativity came
from causal sufficiency, expressions of sufficiency (*ensure*, *guarantee*, *secure*, *make sure*) should favour
negative effects too. They do not: in all four corpora they sit at or above the baseline of other verbs, while *cause*
sits about one point below it.

**Contents:** [What the code does](#what-the-code-does) · [Data](#data) · [Installation](#installation) ·
[Running the analysis](#running-the-analysis) · [Scoring rules](#scoring-rules) ·
[The Glass replication](#the-glass-replication) · [Repository layout](#repository-layout) ·
[Outputs](#outputs) · [Reproducibility notes](#reproducibility-notes) · [References](#references) · [Licence](#licence)

## What the code does

1. **Samples** lines for each expression from four corpora (enTenTen21, BNC, COCA, r/AskReddit).
2. **Parses** each line with spaCy and finds the studied expression.
3. **Finds the effect** the expression introduces (its object, its complement, or, for *result from*, *because of*
   etc., the phrase before it), tags the construction ("frame"), and **scores** the effect word by word with the
   Hedonometer (labMT 2.0: 1 = sad, 9 = happy, 5 = neutral).
4. **Compares** each expression with a baseline of all other lexical verbs in the same corpus, scored by the same
   rules (Welch's t-test, Cohen's d, Holm correction).

Only the sentiment of the effect is analysed.

## Data

Corpus text is **not** included: COCA, the BNC, enTenTen21 and Reddit text cannot be redistributed. To re-run the
analysis, obtain the samples and place them under `data/raw/`:

| Corpus | Source | Sample per expression | Where it goes |
|---|---|---|---|
| enTenTen21 | Sketch Engine, `preloaded/ententen21_tt31` | 10,000 random concordance lines | `data/raw/sketchengine/` |
| BNC | Sketch Engine, `preloaded/bnc2_tt31` | 10,000 random lines, or all hits if fewer | `data/raw/sketchengine/` |
| COCA | english-corpora.org (List search, "ALL FORMS (SAMPLE): 500") | 500 random lines (the site's maximum) | `data/raw/coca/` |
| AskReddit | Pushshift / Arctic Shift dump `RC_2018-01.zst` (e.g. via Academic Torrents) | every January 2018 sentence containing a studied expression | `data/raw/reddit/` |

- [`docs/queries.tsv`](docs/queries.tsv) lists every query used, with the corpus, the number of hits and the sample size.
- [`docs/concordance_format.md`](docs/concordance_format.md) describes the file format the scripts expect.
- Baselines: 10,000 random lexical-verb lines from each of enTenTen21 and BNC (`[tag="VV.*"]`); a random
  3-million-word subsample of AskReddit's first 10 million words of January 2018; for COCA, which offers no random
  sample of all verbs, every lexical verb in 5,500 random lines around 11 frequent function words.

Lexicons go in `lexicons/` (see [`lexicons/README.md`](lexicons/README.md)): the Hedonometer word list
(labMT 2.0 English) and, for one robustness check, the Warriner et al. (2013) valence norms.

## Installation

Python 3.11 (spaCy 3.6.0 supports Python up to 3.11). The parser version matters: the results were produced with
spaCy 3.6.0 and the model `en_core_web_sm` 3.6.0, the version Glass used.

```bash
git clone <this repository> && cd <repository>
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.6.0/en_core_web_sm-3.6.0-py3-none-any.whl
pip install -e .
python -m nltk.downloader punkt_tab wordnet        # sentence splitting and lemmatising for the Reddit steps
```

All paths are set in [`causesent/config.py`](causesent/config.py). By default data live in `data/` inside the
repository; set `CAUSESENT_DATA` to use another folder.

## Running the analysis

[`run_pipeline.sh`](run_pipeline.sh) runs every step in order; each step skips work already done. The steps:

| Step | Script | What it does |
|---|---|---|
| 1 | `pipeline/01_reddit_extract_askreddit.py` | extract the r/AskReddit comments from the monthly dump |
| 2 | `pipeline/02_reddit_sentences.py` | Glass's comment filters and sentence splitting; keep sentences with a studied expression (and, separately, all sentences of the first 10 million words, for the baseline) |
| 3 | `pipeline/03_reddit_score.py` | parse the Reddit sentences, find and score every studied expression |
| 4 | `pipeline/04_reddit_baseline.py` | Reddit baseline: all other verbs in a 3-million-word sample |
| 5 | `pipeline/05_coca_prepare.py` | convert COCA's paragraph marks and copyright gaps into sentence boundaries |
| 6 | `pipeline/06_score_concordances.py` | score the enTenTen21 / BNC (`--source sketchengine`) or COCA (`--source coca`) samples |
| 7 | `pipeline/07_sketchengine_baseline.py` | enTenTen21 and BNC baselines |
| 8 | `pipeline/08_coca_baseline.py` | COCA baseline |
| 9 | `pipeline/09_statistics.py` | every expression against its corpus's baseline, for 4 scoring variants × 3 measures; summary tables by expression and by frame |
| 10 | `pipeline/10_extra_statistics.py` | further analyses: distributions, ranks among other verbs, frame-adjusted means, frame-by-frame comparisons, word groups |
| – | `figures/*.py` | figures (main figure: `fig_effect_sentiment.py`) |

Parsing is the slow part: the Reddit steps parse about 2 million sentences, the concordance steps about 1.7 million
lines. Everything after step 8 runs in minutes.

## Scoring rules

The full rules, with examples, are in [`docs/scoring_rules.md`](docs/scoring_rules.md). In short:

- **Effect after the expression.** No complement: the object's head noun (*cause an accident* → *accident*).
  Object plus complement: the head noun **N** and the complement's verb or adjective **V** (*force the victims to
  flee* → *victim* + *flee*; *made him happy* → *he* + *happy*). That-clauses: the clause's subject and its verb or
  predicate (*ensure that everyone is safe* → *everyone* + *safe*).
- **Effect before the expression** (*result from*, *stem from*, *because of*, *due to* …): its head noun, or the
  subject and verb of the clause (*the game was cancelled because of rain* → *game* + *cancel*).
- **Correlation** (*associated with*, *correlated with*): both sides are scored.
- **Scores.** Each word is a lemma looked up in the Hedonometer. A line scores the mean of N and V (or whichever is
  rated); without a complement, the head noun. An expression's mean pools all its lines and is reported only with at
  least 30 scored lines per corpus.
- **Which lines count.** Passives and negated effects (*persuaded him not to go*) are reported separately; negation
  of the causal word itself (*didn't persuade him*) is kept, as in Glass (2023). *make* and *get* count only in their
  causative frames.
- **Robustness.** Four scoring variants: raw (main); neutral words (4–6) removed; adjectives and noun compounds
  averaged into the noun; the Warriner et al. (2013) valence norms instead of the Hedonometer.
- **Rule switches.** Three refinements (scoring the second noun in *caused me concerns*; looking through quantity
  words, *a lot of questions* → *question*; not scoring pronouns) are available as switches, off by default; see
  [`docs/scoring_rules.md`](docs/scoring_rules.md#8-rule-switches-october-2026).

## The Glass replication

[`glass_replication/`](glass_replication/) re-runs Glass's own code (from her OSF project, <https://osf.io/mv5gk/>),
first on her first 10 million words of r/AskReddit, January 2018 (99.5% of her rows reproduced; *cause*
direct-object mean 3.97 against her 3.96), then on the whole month (4.5 million comments, 139 million words). See its
[README](glass_replication/README.md).

## Repository layout

```
causesent/              the library
  config.py             paths and rule switches
  targets.py            the 80 studied expressions, their kinds, word groups, and the Reddit pre-filter
  frames.py             finding the effect and tagging the frame (the rules)
  scoring.py            scoring variants, which lines count, loading the scored lines
  stats.py              Welch's t-test, Cohen's d, Holm correction
pipeline/               steps 1-10, run in order
figures/                figures from the results
glass_replication/      re-running Glass's (2023) own code on AskReddit
spot_check/             a manual check of 800 random scored COCA lines (10 per expression)
docs/                   scoring rules, the corpus queries, the concordance file format
lexicons/               (download) the Hedonometer and Warriner lexicons
data/                   (created by the pipeline; not in the repository) raw samples, scored lines, results, figures
run_pipeline.sh         the whole analysis in order
```

## Outputs

| File (under `data/`) | Contents |
|---|---|
| `results/pvalues_all.tsv` | the main results: for each scoring × corpus × measure × expression, n, mean, baseline, difference with 95% CI, p, Cohen's d, Holm-corrected p |
| `results/tables/<scoring>_<corpus>_aggregate.tsv` | per expression: lines, mean, % of effects below 5, passive and negated groups |
| `results/tables/<scoring>_<corpus>_by_frame.tsv` | per expression and frame: lines and means of N, V and the line score |
| `results/extra/*.tsv` | the further analyses of step 10 |
| `results/lines_<scoring>.pkl` | every line in the results with its scored words and scores (no corpus text) |
| `work/*_scored/` | every analysed line: frame, scored lemmas, scores, and the first 300 characters of the sentence (corpus text: do not redistribute) |
| `figures/` | figures |

## Reproducibility notes

- spaCy's analyses depend on the model version: use exactly `en_core_web_sm` 3.6.0.
- Re-running the COCA steps with this code gives results that are deterministic but not identical to the results
  first reported for COCA (October 2026): scored-line counts within 2% and expression means within 0.08, because the
  exact code version used then was not preserved. TODO: remove this note if the released results are re-run with
  this code.

## References

Dodds, Peter Sheridan, Kameron Decker Harris, Isabel M. Kloumann, Catherine A. Bliss & Christopher M. Danforth. 2011.
Temporal patterns of happiness and information in a global social network: Hedonometrics and Twitter. *PLoS ONE*
6(12). e26752. <https://doi.org/10.1371/journal.pone.0026752>

Dodds, Peter Sheridan et al. 2015. Human language reveals a universal positivity bias. *Proceedings of the National
Academy of Sciences* 112(8). 2389–2394. <https://doi.org/10.1073/pnas.1411678112>

Glass, Lelia. 2023. Using the Anna Karenina principle to explain why *cause* favors negative-sentiment complements.
*Semantics and Pragmatics* 16(6). 1–48. <https://doi.org/10.3765/sp.16.6>

Honnibal, Matthew, Ines Montani, Sofie Van Landeghem & Adriane Boyd. 2020. spaCy: Industrial-strength natural
language processing in Python. <https://doi.org/10.5281/zenodo.1212303>

Reagan, Andrew J., Brian Tivnan, Jake Ryland Williams, Christopher M. Danforth & Peter Sheridan Dodds. 2017.
Sentiment analysis methods for understanding large-scale texts: a case for using continuum-scored words and word
shift graphs. *EPJ Data Science* 6(28). <https://doi.org/10.1140/epjds/s13688-017-0121-9>

Warriner, Amy Beth, Victor Kuperman & Marc Brysbaert. 2013. Norms of valence, arousal, and dominance for 13,915
English lemmas. *Behavior Research Methods* 45(4). 1191–1207. <https://doi.org/10.3758/s13428-012-0314-x>

## Licence

MIT; see [`LICENSE`](LICENSE).
