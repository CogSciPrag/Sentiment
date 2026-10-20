"""Panel A, whole month: Glass's own per-sentence code (unpatched, run by glass_month_check.py on the month's sentences
containing the 22 verbs that were too rare in the first 10M words) vs the fast re-run (data/work/glass/month).
Same sentences, same ASCII filter as her post-processing; direct-object means per verb.
Writes data/results/glass/month_check.tsv and prints row agreement."""
import glob, csv, os
from collections import Counter
import numpy as np, pandas as pd
from causesent import config
W = config.WORK / "glass"; OUT = config.RESULTS / "glass"; OUT.mkdir(parents=True, exist_ok=True)

VF = ["verb", "trans_intrans", "clause", "subreddit", "clause_hedon", "dobj", "dobj_ont", "nsubj", "nsubj_ont",
      "ccomp", "ccomp_hedon", "nsubj_hedon", "dobj_hedon"]
FOCUS = os.environ.get("FOCUS", "").split(",")
EX = set("’‘“”´—-–―é:.…\t\n")
keep = lambda s: all(ord(c) < 128 for c in str(s) if c not in EX)
rd = lambda files: pd.concat([pd.read_csv(f, sep="\t", names=VF, header=None, quoting=csv.QUOTE_MINIMAL, low_memory=False) for f in files])
H = rd(sorted(glob.glob(str(W / "month_check" / "parts" / "*.verb.tsv")))); O = rd(sorted(glob.glob(str(W / "month" / "parts" / "*.verb.tsv"))))
H = H[H.verb.isin(FOCUS) & H.clause.map(keep)]; O = O[O.verb.isin(FOCUS) & O.clause.map(keep)]
hc, oc = Counter(zip(H.verb, H.clause)), Counter(zip(O.verb, O.clause)); m = sum((hc & oc).values())
print(f"rows: Glass code {len(H):,}, ours {len(O):,}; identical verb+clause rows {m:,} ({m/max(len(H),1):.2%} of hers)")
rows = []
for v in FOCUS:
    h = pd.to_numeric(H[H.verb == v].dobj_hedon, errors="coerce").dropna(); o = pd.to_numeric(O[O.verb == v].dobj_hedon, errors="coerce").dropna()
    rows.append(dict(verb=v, dobj_n_Glass=len(h), dobj_mean_Glass=h.mean() if len(h) else np.nan, dobj_n_ours=len(o), dobj_mean_ours=o.mean() if len(o) else np.nan))
R = pd.DataFrame(rows); R.to_csv(OUT / "month_check.tsv", sep="\t", index=False, float_format="%.4f")
open(OUT / "month_check_rows.txt", "w").write(f"{len(H)}\t{len(O)}\t{m}\n")
print(R.round(3).to_string(index=False))
