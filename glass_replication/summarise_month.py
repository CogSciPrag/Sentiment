"""Per-verb summary of Glass's measures on the full-month target sentences (works on finished parts too).
Filters as in her postprocessing: verb pmw >= 2 (month counts), ASCII-only clauses.
Baseline = 'all other verbs' in the validated 10M-word run (data/work/glass/val10M), as in her analysis.
Minimum: a mean is reported only with >= 30 scored instances."""
import glob, os, csv, json, sys
import pandas as pd, numpy as np
import glass_pipeline_fast as F
from causesent import config
W = config.WORK; OUT = config.RESULTS / "glass"; OUT.mkdir(parents=True, exist_ok=True)
from targets import LEMMAS

MIN_N = 30
parts = sorted(glob.glob(str(W / "glass" / "month" / "parts" / "*.verb.tsv")))
D = pd.concat([pd.read_csv(p, sep="\t", names=F.VF, header=None, quoting=csv.QUOTE_MINIMAL, low_memory=False) for p in parts])
stats = json.load(open(W / "glass_sentences_month" / "pass1_stats.txt"))
counts = {}
for raw in open(W / "glass_sentences_month" / "word_counts.tsv", "rb"):
    w, c = raw.decode("utf-8")[:-1].rsplit("\t", 1); counts[w] = int(c)
D["verb_pmw"] = D.verb.map(lambda v: counts.get(v, 0) * 1e6 / stats["words"])
D = D[D.verb_pmw >= 2]
EX = set("’‘“”´—-–―é:.…\t\n")
D = D[D.clause.astype(str).map(lambda s: all(ord(c) < 128 for c in s if c not in EX))]
D = D[D.verb.isin(LEMMAS)]
B = pd.read_csv(W / "glass" / "val10M" / "verb_clause_df.csv", index_col=0, low_memory=False)
if os.environ.get("BAND"):   # optional: apply Glass's 4-6 exclusion to the single-word object/subject scores too
    for X in (D, B):
        for c in ["dobj_hedon", "nsubj_hedon"]:
            X.loc[X[c].between(4, 6), c] = np.nan

def row(g, name):
    r = {"verb": name, "clauses": len(g)}
    for c in ["dobj", "ccomp", "clause", "nsubj"]:
        x = g[c + "_hedon"].dropna()
        r[c + "_n"] = len(x)
        r[c + "_mean"] = x.mean() if len(x) >= MIN_N else np.nan
        r[c + "_se"] = x.sem() if len(x) >= MIN_N else np.nan
        r[c + "_pct_below5"] = (x < 5).mean() * 100 if len(x) >= MIN_N else np.nan
    return r
rows = [row(g, v) for v, g in D.groupby("verb")]
rows.append(row(B[~B.verb.isin(["cause"])], "BASELINE: all other verbs (first 10M words)"))
S = pd.DataFrame(rows).sort_values("dobj_mean")
S.to_csv(sys.argv[1] if len(sys.argv) > 1 else OUT / "month_summary.tsv", sep="\t", index=False, float_format="%.3f")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
print(f"{len(parts)} parts, {len(D):,} target-verb clauses")
print(S[["verb", "clauses", "dobj_n", "dobj_mean", "dobj_pct_below5", "ccomp_n", "ccomp_mean", "clause_mean", "nsubj_n", "nsubj_mean"]].round(2).to_string(index=False))
