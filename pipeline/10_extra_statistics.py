"""Step 10: extra analyses on the scored lines (AskReddit is the reference corpus for the baseline comparisons here).

  word_stats.tsv              per expression, corpus and scoring: mean, distribution (% below 4 / 4-6 / above 6), rank
                              among the 80 expressions, a frame-adjusted mean (the expression's frame means weighted by
                              the baseline's mix of frames), and for AskReddit the comparison with the baseline (Welch,
                              Cohen's d, Holm) and the share of other verbs whose mean is lower
  frame_by_frame_vs_baseline.tsv   AskReddit: each expression vs the baseline within the same frame
  other_verb_means_<scoring>.tsv   AskReddit: mean per baseline verb (verbs with at least 30 lines)
  group_means.tsv             mean of the expressions' means per word group, with a bootstrap 95% CI over expressions
Input: data/results/lines_<scoring>.pkl (step 9). Output: data/results/extra/ (and extra/all_lines.pkl for the figures).
Usage: python pipeline/10_extra_statistics.py
"""
import os
import numpy as np, pandas as pd
from causesent import config, scoring as SL
from causesent.stats import welch, holm
from causesent.targets import ITEMS, PREVENTION, EFFECT_LEFT, WORD_GROUPS

MIN_N = config.MIN_N
OUT = str(config.RESULTS / "extra"); os.makedirs(OUT, exist_ok=True)

# one table with the line scores of all four scoring variants side by side
A = None
for sc in ["raw", "band", "war", "adj"]:
    L = pd.read_pickle(config.RESULTS / f"lines_{sc}.pkl").reset_index(drop=True)
    if A is None:
        A = L[["corpus", "item", "frame", "verb", "N_words", "V_words", "N_mods"]].copy()
    else:   # the four files list the same lines in the same order
        assert len(L) == len(A) and (L["item"].values == A["item"].values).all() and (L["frame"].values == A["frame"].values).all()
    A[f"N_{sc}"] = L.score_N.values; A[f"V_{sc}"] = L.score_V.values; A[f"agg_{sc}"] = L.agg_score.values
A.to_pickle(os.path.join(OUT, "all_lines.pkl"))
B = A[(A.item == SL.BASELINE) & (A.corpus == "reddit")]
T = A[A.item != SL.BASELINE]

grp_of = {w: g for g, ws in WORD_GROUPS.items() for w in ws}
missing = [w for w in ITEMS if w not in grp_of]; assert not missing, missing

rows, frows = [], []
for sc in ["raw", "band", "war", "adj"]:
    col = f"agg_{sc}"
    b = B[col].dropna()
    bfreq = B[B[col].notna()].frame.value_counts(normalize=True)
    bframe = B.groupby("frame")[col].agg(["count", "mean"])
    vm = B.groupby("verb")[col].agg(["count", "mean"]); vm = vm[vm["count"] >= MIN_N]["mean"]   # per-verb means (other verbs)
    for corp in [c for c in SL.CORPORA if c in set(T.corpus)]:
        C = T[T.corpus == corp]
        allmeans = C.groupby("item")[col].agg(["count", "mean"]); allmeans = allmeans[allmeans["count"] >= MIN_N]["mean"]
        for item, g in C.groupby("item"):
            x = g[col].dropna()
            if len(x) < MIN_N: continue
            r = dict(scoring=sc, corpus=corp, item=item, group=grp_of[item], n=len(x), mean=x.mean(),
                     pct_below4=(x < 4).mean() * 100, pct_4to6=((x >= 4) & (x <= 6)).mean() * 100, pct_above6=(x > 6).mean() * 100,
                     rank_among_items=int((allmeans < x.mean()).sum()) + 1, n_words_ranked=len(allmeans))
            # frame-adjusted: weight the word's frame means by the baseline's frame mix (frames with >= 10 lines)
            fm = g.groupby("frame")[col].agg(["count", "mean"]); fm = fm[fm["count"] >= 10]
            w = bfreq.reindex(fm.index).fillna(0)
            r["frame_adjusted_mean"] = float((fm["mean"] * w).sum() / w.sum()) if w.sum() > 0 else np.nan
            r["baseline_frame_share_covered"] = float(w.sum())
            if corp == "reddit":
                d, lo, hi, p, dz = welch(x, b)
                r.update(baseline=b.mean(), diff=d, ci_low=lo, ci_high=hi, p=p, cohens_d=dz,
                         pct_verbs_below=(vm < x.mean()).mean() * 100, n_verbs=len(vm))
                for fr, gf in g.groupby("frame"):
                    xf = gf[col].dropna(); bf = B[B.frame == fr][col].dropna()
                    if len(xf) >= MIN_N and len(bf) >= MIN_N:
                        d2, lo2, hi2, p2, dz2 = welch(xf, bf)
                        frows.append(dict(scoring=sc, item=item, frame=fr, n=len(xf), mean=xf.mean(), baseline_frame_mean=bf.mean(),
                                          baseline_n=len(bf), diff=d2, ci_low=lo2, ci_high=hi2, cohens_d=dz2, p=p2))
            rows.append(r)
S = pd.DataFrame(rows)
# Holm correction within each scoring (Reddit comparisons)
S["p_holm"] = np.nan
for sc in ["raw", "band", "war", "adj"]:
    m = (S.scoring == sc) & S.p.notna()
    if m.any(): S.loc[m, "p_holm"] = holm(S.loc[m, "p"].values)
S.to_csv(OUT + "/word_stats.tsv", sep="\t", index=False, float_format="%.4f")
F = pd.DataFrame(frows); F.to_csv(OUT + "/frame_by_frame_vs_baseline.tsv", sep="\t", index=False, float_format="%.4f")

# per-verb means of other verbs (for the ranking figure)
for sc in ["raw", "band", "war", "adj"]:
    vm = B.groupby("verb")[f"agg_{sc}"].agg(["count", "mean"]); vm = vm[vm["count"] >= MIN_N]
    vm.to_csv(f"{OUT}/other_verb_means_{sc}.tsv", sep="\t", float_format="%.4f")

# groups: unweighted mean of word means, with a bootstrap CI over words
gr = []
rng = np.random.default_rng(1)
for (sc, corp), s in S.groupby(["scoring", "corpus"]):
    for gname, gs in s.groupby("group"):
        v = gs["mean"].values
        bs = [rng.choice(v, len(v)).mean() for _ in range(2000)] if len(v) > 1 else [v[0]]
        gr.append(dict(scoring=sc, corpus=corp, group=gname, words=len(v), mean_of_word_means=v.mean(),
                       ci_low=np.percentile(bs, 2.5), ci_high=np.percentile(bs, 97.5), words_list=", ".join(gs.item)))
Gd = pd.DataFrame(gr); Gd.to_csv(OUT + "/group_means.tsv", sep="\t", index=False, float_format="%.4f")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
if len(S) and (S.corpus == "reddit").any():
    k = S[(S.corpus == "reddit") & (S.scoring == "raw")].set_index("item")
    print(k.reindex(["cause", "make", "ensure", "guarantee", "secure", "prevent", "lead to"])[
        ["n", "mean", "diff", "cohens_d", "p_holm", "pct_verbs_below", "frame_adjusted_mean"]].round(3))
print("done:", len(S), "word rows,", len(F), "frame-by-frame rows,", len(Gd), "group rows; written to", OUT)
