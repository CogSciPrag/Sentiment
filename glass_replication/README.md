# Replicating Glass (2023) on AskReddit

This folder re-runs Lelia Glass's own analysis code from her OSF project (<https://osf.io/mv5gk/>, notebook
`GetCausationClausesFromReddit_updated2023.ipynb`), first on her own sample and then on the whole month.

- `glass_pipeline.py`: her functions (`get_clauses`, `get_verb`, `get_sconj`, the Hedonometer lookups and her
  post-filters), copied verbatim. Changes that leave the output unchanged: the sentence scorer uses spaCy's tokenizer
  alone (same tokens), and the WordNet category columns are left blank (they play no part in the sentiment scores).
- `glass_pipeline_fast.py`: runs her per-sentence code in parallel, re-using each parse (spaCy's output for a given
  string is deterministic).
- `targets.py`: the verbs of the replication and their word forms (used to pre-select sentences for the month).

## Steps

Run from the repository root. Her code needs spaCy `en_core_web_sm` 3.6.0, the version her notebook installs.

```bash
# 1. sentences: her first 10 million words (all sentences), and the whole month (sentences with a target verb)
python pipeline/02_reddit_sentences.py reddit_sentences_10M --select all --max-words 10000000
python pipeline/02_reddit_sentences.py glass_sentences_month --select glass
# 2. her extraction on both (writes her output files: verb_clause_df.csv, reddit_sconj_sentiment.txt, ...)
python glass_replication/glass_pass2_targets.py data/work/reddit_sentences_10M data/work/glass/val10M 2
python glass_replication/glass_pass2_targets.py data/work/glass_sentences_month data/work/glass/month 2
# 3. validation against her posted output (download verb_clause_df.csv from her OSF project into data/raw/glass_osf/)
python glass_replication/compare_val.py
# 4. per-verb summary of her measures on the whole month (direct object, clausal complement, clause, subject)
python glass_replication/summarise_month.py
BAND=1 python glass_replication/summarise_month.py data/results/glass/month_summary_band.tsv   # neutral band removed
# 5. optional: her unmodified code on the month's sentences for the verbs too rare in 10M words, and the comparison
FOCUS=arise,compel,contribute,depend,enable,ensure,facilitate,foster,generate,hinder,induce,influence,inhibit,necessitate,permit,prompt,provoke,render,result,secure,spark,stem \
  python glass_replication/glass_month_check.py data/work/glass_sentences_month data/work/glass/month_check 2
FOCUS=... python glass_replication/compare_month_check.py
# 6. statistics for her measure and the figures
python glass_replication/fig_replication.py raw
python glass_replication/plot_month.py
```

## Results (January 2018)

- Her first 10 million words: 436,218 rows against her 436,110, 99.5% identical; *cause* direct-object mean 3.97
  against her 3.96; all other verbs 5.65 in both. Rank order of the verbs' object means: Spearman 0.999.
- Whole-month check (22 less frequent verbs): 18,024 of her code's 18,024 rows identical to the fast re-run.
- Whole month (4,524,771 comments after her filters, 138.8 million words): *cause* has the most negative direct
  objects of the 37 verbs (3.91, n = 4,063; 65% below 5, against 16% for other verbs); *ensure* 5.91, *guarantee* 5.72
  and *secure* 5.72 are at or above the baseline (5.65).
