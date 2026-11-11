"""Step 9: compare every studied expression with its corpus's baseline, for each scoring variant and measure.

For each scoring variant (raw, band, adj, war; see causesent/scoring.py), corpus and measure
  NV  the line score (N and V averaged when the effect has a complement, else N)
  N   the head noun only
  V   the verb / adjective / predicate only
each expression's lines are compared with all other verbs in the same corpus: Welch's t-test, 95% CI of the
difference, Cohen's d, and Holm correction within each scoring x corpus x measure x polarity. A test is run only with
at least 30 scored lines on both sides.
Polarity (decided 1 Nov 2026): every comparison is made three times, for the positive forms of the word ('caused'),
its negative forms ('didn't cause', 'nothing caused') and both combined, each against the same split of the baseline
(positive vs positive baseline lines, etc.). 'combined' is the headline (and what the figures use).
Baselines (column 'baseline_type'): every expression is compared with the baseline of all other verbs ('verbs'). The
connectives (targets.CONNECTIVES) are also compared with Glass's (2023) baseline of all other subordinating
conjunctions ('conjunctions'; decided 1 Nov 2026), their main comparison. The noun items (targets.NOUN_ITEMS: the
cause of, the effect of, result of, consequence of, reason) are also compared with all other nouns in 'the N of X' and
'X is the N of Y' ('nouns'; decided 1 Nov 2026, added 2 Nov 2026). Holm correction within scoring x corpus x
measure x polarity x baseline_type. The figures use the verb baseline.

Outputs in data/results/:
  pvalues_all.tsv                      the comparisons above (the main results table)
  lines_<scoring>.pkl                  every line that enters the results, with its scores (used by the figures)
  tables/<scoring>_<corpus>_aggregate.tsv   per expression: scored lines, mean, % below 5 (combined), n and mean of
                                       the positive and negative forms, passive and negated-effect groups
  tables/<scoring>_<corpus>_by_frame.tsv    per expression and frame: lines and means of N, V, N+V and the line score
Usage:  python pipeline/09_statistics.py [scoring ...]
"""
import gc, sys
import numpy as np, pandas as pd
from causesent import config, scoring as SL
from causesent.stats import welch, holm, stars
from causesent.targets import ITEMS, PREVENTION, EFFECT_LEFT, CONNECTIVES, NOUN_ITEMS

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
        if "obj_forms" in D: D.loc[wf, "N_forms"] = D.loc[wf, "obj_forms"]
        for c in ["score_N", "score_NV", "agg_score"]: D.loc[wf, c] = D.loc[wf, "score_obj"]
    rows = []
    if "polarity" not in D: D["polarity"] = ""
    for (item, frame, group), g in D[D.frame != "no effect"].groupby(["item", "frame", "group"]):
        r = dict(corpus=corpus, item=item, frame=frame, group=group, lines=len(g),
                 lines_positive=int((g.polarity == "positive").sum()), lines_negative=int((g.polarity == "negative").sum()))
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
        for pol in ["positive", "negative"]:
            r[pol + "_n"], r[pol + "_mean"] = _mean(main[main.polarity == pol].agg_score)
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
    for D in (T, B):    # the strings actually looked up (form-first rule), for the word-shift figures
        D["N_keys"] = SL.scored_keys(D, "N_words", "N_forms"); D["V_keys"] = SL.scored_keys(D, "V_words", "V_forms")
    keep = ["corpus", "item", "frame", "verb", "N_words", "V_words", "N_mods", "score_N", "score_V", "agg_score",
            "prevention", "effect_left", "removal", "polarity", "N_keys", "V_keys"]
    pd.concat([T, B], ignore_index=True)[[c for c in keep if c in T.columns or c in B.columns]].to_pickle(
        config.RESULTS / f"lines_{sc}.pkl")
    for corp in corpora:
        Tc, Bcorp = T[T.corpus == corp], B[B.corpus == corp]
        for btype, Bc, items in [("verbs", Bcorp[Bcorp.item == SL.BASELINE], None),
                                 ("conjunctions", Bcorp[Bcorp.item == SL.BASELINE_SCONJ], CONNECTIVES),
                                 ("nouns", Bcorp[Bcorp.item == SL.BASELINE_NOUN], NOUN_ITEMS)]:
          for (pol, Tp), (_, Bp) in zip(SL.by_polarity(Tc), SL.by_polarity(Bc)):
            for meas, col in [("NV", "agg_score"), ("N", "score_N"), ("V", "score_V")]:
              b = Bp[col].dropna()
              for item, g in Tc.groupby("item"):
                if items is not None and item not in items: continue
                x = Tp.loc[Tp.item == item, col].dropna()
                r = dict(scoring=sc, corpus=corp, measure=meas, polarity=pol, baseline_type=btype, item=item, n=len(x),
                         mean=x.mean() if len(x) else np.nan, baseline_n=len(b), baseline=b.mean() if len(b) else np.nan)
                if len(x) >= MIN_N and len(b) >= MIN_N:
                    d, lo, hi, p, dz = welch(x, b); r.update(diff=d, ci_low=lo, ci_high=hi, p=p, cohens_d=dz)
                rows.append(r)
    # Free this scoring's data before the next one is built (2 Nov 2026): otherwise two scorings' copies of the ~5M
    # records are held at once, and the 101-word run was killed for lack of memory (8 GB) during the second scoring.
    for _v in ("R", "T", "B", "D", "Tc", "Bcorp", "Bc", "Tp", "Bp", "b", "g", "x"):
        globals().pop(_v, None)
    gc.collect()
P = pd.DataFrame(rows)
P["p_holm"] = np.nan
for _, idx in P[P.p.notna()].groupby(["scoring", "corpus", "measure", "polarity", "baseline_type"]).groups.items():
    P.loc[idx, "p_holm"] = holm(P.loc[idx, "p"].values)
P["stars"] = P.p_holm.map(stars)
P["rules"] = config.rules_label()
P.to_csv(config.RESULTS / "pvalues_all.tsv", sep="\t", index=False, float_format="%.6g")
pd.set_option("display.width", 200)
show = P[(P.scoring == scorings[0]) & (P.measure == "NV") & (P.polarity == "combined") &
         P.item.isin(["cause", "ensure", "guarantee", "secure", "because (clause)", "because of"])]
print(show[["corpus", "item", "baseline_type", "n", "mean", "baseline", "cohens_d", "stars"]].round(3).to_string(index=False))
print("done:", len(P), "comparisons written to", config.RESULTS / "pvalues_all.tsv", flush=True)
