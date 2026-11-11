"""Step 11: the make / get follow-up (version C of their two-object lines; decided 30 Oct 2026).

Headline results (steps 9 and 10) count no make or get line with two objects (version B). This step compares, for
make and get in each corpus and scoring variant, version B with version C:
  make  N + N lines count unless the second noun is a created object ('made him a sandwich', CREATED_OBJECTS)
  get   N + N lines count only when the subject of 'get' is clearly not a person ('having a degree won't get you
        a job'; see scoring.subject_not_person)
Each version is compared with the corpus's baseline (Welch's t-test, Cohen's d), and the two versions' means are
given side by side.

Outputs in data/results/extra/:
  make_get_versions.tsv     per scoring, corpus, item and version: n, mean, baseline, difference, CI, p, Cohen's d
  make_get_nn_lines.tsv     every make / get N + N line (main group), with the second noun, the subject and whether it
                            counts in version C, for checking the word lists in causesent/scoring.py
Usage: python pipeline/11_make_get_followup.py [scoring ...]      (default: raw)
"""
import os, sys
import numpy as np, pandas as pd
from causesent import config, scoring as SL
from causesent.stats import welch

scorings = sys.argv[1:] or ["raw"]
OUT = config.RESULTS / "extra"; os.makedirs(OUT, exist_ok=True)
T0, B0 = SL.load_records()
corpora = [c for c in SL.CORPORA if c in set(T0.corpus)]
print("rules:", config.rules_label(), "| corpora:", ", ".join(corpora), flush=True)

M = T0[T0.item.isin(["make", "get"]) & (T0.group == "main") & T0.frame.isin(SL.NN_FRAMES)].copy()
M["counts_in_C"] = SL.nn_followup_keep(M)
M[["corpus", "item", "frame", "N_words", "subj_words", "subj_pos", "counts_in_C"]].to_csv(
    OUT / "make_get_nn_lines.tsv", sep="\t", index=False)

rows = []
for sc in scorings:
    R = SL.rescore(T0, sc); B = SL.prep(SL.rescore(B0, sc), baseline=True)
    for version in ("B", "C"):
        T = SL.prep(R, make_get_nn=version)
        for corp in corpora:
            b = B[B.corpus == corp].agg_score.dropna()
            for item in ("make", "get"):
                g = T[(T.corpus == corp) & (T.item == item)]
                x = g.agg_score.dropna()
                r = dict(scoring=sc, corpus=corp, item=item, version=version, n=len(x),
                         n_two_objects=int(g.frame.isin(SL.NN_FRAMES).sum()),
                         mean=x.mean() if len(x) else np.nan, baseline=b.mean(), baseline_n=len(b))
                if len(x) >= config.MIN_N and len(b) >= config.MIN_N:
                    d, lo, hi, p, dz = welch(x, b); r.update(diff=d, ci_low=lo, ci_high=hi, p=p, cohens_d=dz)
                rows.append(r)
P = pd.DataFrame(rows); P["rules"] = config.rules_label()
P.to_csv(OUT / "make_get_versions.tsv", sep="\t", index=False, float_format="%.6g")
pd.set_option("display.width", 200)
print(P[["scoring", "corpus", "item", "version", "n", "n_two_objects", "mean", "baseline", "cohens_d"]].round(3).to_string(index=False))
print("done:", OUT / "make_get_versions.tsv", flush=True)
