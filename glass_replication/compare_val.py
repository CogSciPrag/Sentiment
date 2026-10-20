"""Validation: our re-run of Glass's pipeline on the first 10M AskReddit words vs her posted verb_clause_df.csv.
(1) row-level agreement (verb + clause text, in order); (2) her headline numbers per verb."""
import pandas as pd, numpy as np, difflib, sys
from causesent import config
OUT = config.RESULTS / "glass"; OUT.mkdir(parents=True, exist_ok=True)
from targets import LEMMAS

H = pd.read_csv(config.RAW / "glass_osf" / "verb_clause_df.csv", index_col=0, low_memory=False)
O = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else config.WORK / "glass" / "val10M" / "verb_clause_df.csv", index_col=0, low_memory=False)

from collections import Counter
hc = Counter(zip(H.verb, H.clause)); oc = Counter(zip(O.verb, O.clause))
m = sum((hc & oc).values())
print(f"rows: hers {len(H):,}, ours {len(O):,}; identical verb+clause rows: {m:,} "
      f"({m/len(H):.1%} of hers, {m/len(O):.1%} of ours)")
print(f"distinct verbs: hers {H.verb.nunique():,}, ours {O.verb.nunique():,}")

def summ(D, label):
    rows = []
    for v in LEMMAS + ["ALL OTHER VERBS"]:
        g = D[D.verb == v] if v != "ALL OTHER VERBS" else D[~D.verb.isin(["cause"])]
        r = {"verb": v, "src": label, "clauses": len(g)}
        for c in ["dobj_hedon", "ccomp_hedon", "clause_hedon", "nsubj_hedon"]:
            x = g[c].dropna()
            r[c.replace("_hedon", "") + "_n"] = len(x)
            r[c.replace("_hedon", "") + "_mean"] = x.mean() if len(x) else np.nan
        x = g["dobj_hedon"].dropna(); r["dobj_pct_below5"] = (x < 5).mean() * 100 if len(x) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)

S = pd.concat([summ(H, "Glass"), summ(O, "ours")])
W = S.pivot(index="verb", columns="src")
cols = [(c, s) for c in ["clauses", "dobj_n", "dobj_mean", "dobj_pct_below5", "ccomp_n", "ccomp_mean", "clause_mean"] for s in ["Glass", "ours"]]
W = W[cols].reindex(LEMMAS + ["ALL OTHER VERBS"])
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
print(W.round(2).to_string())
W.columns = [f"{a}_{b}" for a, b in W.columns]
W.to_csv(OUT / "validation_10M_glass_vs_ours.tsv", sep="\t", float_format="%.3f")
print("\nSpearman (dobj mean, verbs with >=30 objects in both):")
ok_ = W[(W.dobj_n_Glass >= 30) & (W.dobj_n_ours >= 30)]
print(len(ok_), round(ok_[["dobj_mean_Glass", "dobj_mean_ours"]].corr(method="spearman").iloc[0, 1], 3))
