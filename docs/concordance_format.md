# Format of the concordance samples

Steps 5–8 read concordance samples (random concordance lines for one query) as CSV, in the layout of Sketch Engine's
CSV export: a few metadata rows, a header row starting with `Reference`, then one row per line:

```
"corpus","preloaded/bnc2_tt31"
"item","cause"
"query","[lemma=""cause"" & tag=""V.*""]"
"hits_in_corpus","20078"
"sample","10000"
"context","truncated to 100 characters each side"
"Reference","Left","KWIC","Right"
"J2B","... made significant progress in their search for the exact location of the gene for","causing","the disease ..."
```

- `item` must be one of the studied expressions in `causesent/targets.py` (exactly as spelled there, e.g.
  `because (clause)`); for older files without an `item` row, the expression is taken from the file name
  (`se_<corpus>__<expression>.csv`, with `_` for spaces).
- Sketch Engine marks sentence boundaries in the context with `</s><s>`; only the sentence containing the hit is parsed.
- Several samples can be stored in one file, each preceded by a line `### FILE <sample name>` (the sample name is used
  for the output file name). Baseline samples have `BASELINE` in their name (COCA) or are stored in
  `se_baseline_other_verbs.txt` (Sketch Engine).
- COCA samples (english-corpora.org) were saved in the same layout, with the source information
  (`2019 SPOK CBS_Morning #5001936`) in the `Reference` column; step 5 converts COCA's `#` paragraph marks and `@`
  copyright gaps into sentence boundaries.

`docs/queries.tsv` lists every sample used: corpus, expression, query (Sketch Engine CQL or COCA syntax), the number
of hits in the corpus, the sample size and the context width.
