"""Step 9: compare every studied expression with its corpus's baseline, for each scoring variant and measure.

For each scoring variant (raw, band, adj, war; see causesent/scoring.py), corpus and measure
  NV  the line score (N and V averaged when the effect has a complement, else N)
  N   the head noun only
  V   the verb / adjective / predicate only
each expression's lines are compared with all other verbs in the same corpus: Welch's t-test, 95% CI of the
difference, Cohen's d, and Holm correction within each scoring x corpus x measure. A test is run only with at least
30 scored lines on both sides.

Outputs in data/results/:
  pvalues_all.tsv                      the comparisons above (the main results table)
  lines_<scoring>.pkl                  every line that enters the results, with its scores (used by the figures)
  tables/<scoring>_<corpus>_aggregate.tsv   per expression: scored lines, mean, % below 5, passive and negated groups
  tables/<scoring>_<corpus>_by_frame.tsv    per expression and frame: lines and means of N, V, N+V and the line score
Usage:  python pipeline/09_statistics.py [scoring ...]
"""
import sys
import numpy as np, pandas as pd
from causesent import config, scoring as SL
from causesent.stats import welch, holm, stars
from causesent.targets import ITEMS, PREVENTION, EFFECT_LEFT

MIN_N = config.MIN_N
scorings = sys.argv[1:] or SL.SCORINGS
config.ensure_dirs(); (config.RESULTS / "tables").mkdir(exist_ok=True)


def _mean(x):
    x = x.dropna()
    return len(x), (x.mean() if len(x) >= MIN_N else np.nan)


def summary_tables(D, corpus, sc):
    """By-frame and aggregate tables for one corpus (all groups, so passives and negated effects are shown too)."""
    D = D[D.item.isin(ITEMS) & (D.frame != "not matched")].copy()
    if "score_obj" in D:            # with/for-N counts only for blame and credit; other verbs keep their direct object
        wf = (D.frame == "V + N + with/for N") & ~D.item.isin(["blame", "credit"])
        D.loc[wf, "frame"] = "V + N"; D.loc[wf, "N_words"] = D.loc[wf, "obj_words"]
        for c in ["score_N", "score_NV", "agg_score"]: D.loc[wf, c] = D.loc[wf, "score_obj"]
    rows = []
    for (item, frame, group), g in D[D.frame != "no effect"].groupby(["item", "frame", "group"]):
        r = dict(corpus=corpus, item=item, frame=frame, group=group, lines=len(g))
        for c in ["score_N", "score_V", "score_NV", "agg_score"]:
            r[c + "_n"], r[c + "_mean"] = _mean(g[c])
        rows.append(r)
    pd.DataFrame(rows).sort_values(["item", "group", "lines"], ascending=[True, True, False]).to_csv(
        config.RESULTS / "tables" / f"{sc}_{corpus}_by_frame.tsv", sep="\t", index=False, float_format="%.3f")
    agg = []
    for item in ITEMS:
        g = D[D.item == item]
        if not len(g): continue
        main = g[(g.group == "main") & (g.frame != "no effect")]
        if item in SL.AGG_FRAMES: main = main[main.frame.isin(SL.AGG_FRAMES[item])]
        n, mu = _mean(main.agg_score)
        r = dict(corpus=corpus, item=item, prevention=item in PREVENTION, effect_left=item in EFFECT_LEFT,
                 lines_total=len(g), lines_no_effect=int((g.frame == "no effect").sum()), n=n, mean=mu,
                 pct_below5=(main.agg_score.dropna() < 5).mean() * 100 if n >= MIN_N else np.nan)
        for grp in ["passive", "negated"]:
            r[grp + "_n"], r[grp + "_mean"] = _mean(g[g.group == grp].agg_score)
        agg.append(r)
    pd.DataFrame(agg).sort_values("mean").to_csv(config.RESULTS / "tables" / f"{sc}_{corpus}_aggregate.tsv", sep="\t",
                                                  index=False, float_format="%.3f")


T0, B0 = SL.load_records()
corpora = [c for c in SL.CORPORA if c in set(T0.corpus)]
print("rules:", config.rules_label(), "| corpora:", ", ".join(corpora), flush=True)
rows = []
for sc in scorings:
    print("scoring", sc, flush=True)
    R = SL.rescore(T0, sc)
    for corp in corpora:
        summary_tables(R[R.corpus == corp], corp, sc)
    T = SL.prep(R); B = SL.prep(SL.rescore(B0, sc), baseline=True)
    keep = ["corpus", "item", "frame", "verb", "N_words", "V_words", "N_mods", "score_N", "score_V", "agg_score",
            "prevention", "effect_left"]
    pd.concat([T, B], ignore_index=True)[[c for c in keep if c in T.columns or c in B.columns]].to_pickle(
        config.RESULTS / f"lines_{sc}.pkl")
    for corp in corpora:
        Tc, Bc = T[T.corpus == corp], B[B.corpus == corp]
        for meas, col in [("NV", "agg_score"), ("N", "score_N"), ("V", "score_V")]:
            b = Bc[col].dropna()
            for item, g in Tc.groupby("item"):
                x = g[col].dropna()
                r = dict(scoring=sc, corpus=corp, measure=meas, item=item, n=len(x), mean=x.mean() if len(x) else np.nan,
                         baseline_n=len(b), baseline=b.mean())
                if len(x) >= MIN_N and len(b) >= MIN_N:
                    d, lo, hi, p, dz = welch(x, b); r.update(diff=d, ci_low=lo, ci_high=hi, p=p, cohens_d=dz)
                rows.append(r)
P = pd.DataFrame(rows)
P["p_holm"] = np.nan
for _, idx in P[P.p.notna()].groupby(["scoring", "corpus", "measure"]).groups.items():
    P.loc[idx, "p_holm"] = holm(P.loc[idx, "p"].values)
P["stars"] = P.p_holm.map(stars)
P["rules"] = config.rules_label()
P.to_csv(config.RESULTS / "pvalues_all.tsv", sep="\t", index=False, float_format="%.6g")
pd.set_option("display.width", 200)
show = P[(P.scoring == scorings[0]) & (P.measure == "NV") & P.item.isin(["cause", "ensure", "guarantee", "secure"])]
print(show[["corpus", "item", "n", "mean", "baseline", "cohens_d", "stars"]].round(3).to_string(index=False))
print("done:", len(P), "comparisons written to", config.RESULTS / "pvalues_all.tsv", flush=True)
