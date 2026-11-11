import numpy as np, pandas as pd
from scipy import stats
from causesent import config
P = pd.read_csv(config.RESULTS / "pvalues_all.tsv", sep="\t")
for sc in ["raw", "band", "war"]:
    Q = P[(P.scoring == sc) & (P.measure == "NV") & (P.polarity == "combined") & (P.baseline_type == "verbs") & (P.n >= 30)]
    W = Q.pivot(index="item", columns="corpus", values="mean"); nc = W.notna().sum(1); m = W.mean(1)
    full = m[nc == 4]; others = full.drop("cause")
    print(f"\n===== {sc}: {len(full)} expressions with n >= 30 in all 4 corpora ({(nc < 4).sum()} with fewer: {', '.join(f'{i} ({nc[i]})' for i in nc[nc < 4].index)})")
    print("lowest 8 (four-corpus mean):"); print(full.sort_values().head(8).round(3).to_string())
    x = full["cause"]; s = others.values; n = len(s)
    t = (x - s.mean()) / (s.std(ddof=1) * np.sqrt((n + 1) / n))
    med, mad = np.median(s), stats.median_abs_deviation(s, scale="normal")
    print(f"cause {x:.3f}; others mean {s.mean():.3f} (SD {s.std(ddof=1):.3f}, median {med:.3f}); gap to next lowest ({others.idxmin()} {others.min():.3f}) = {x - others.min():+.3f}")
    print(f"Crawford-Howell t({n - 1}) = {t:.2f}, p = {2 * stats.t.sf(abs(t), n - 1):.1e}; robust z (median/MAD) = {(x - med) / mad:.1f}; "
          f"rank {int(full.rank()['cause'])} of {len(full)}")
    low = full.sort_values()
    print("next lowest are", (low.iloc[1:4] - x).round(3).to_dict(), "points above cause")
    print("per corpus (all expressions with n >= 30 in that corpus):")
    for c in ["ententen21", "bnc", "reddit", "coca"]:
        col = W[c].dropna(); o = col.drop("cause")
        print(f"   {c:11s} cause {col['cause']:.3f}; rank {int(col.rank()['cause'])}/{len(col)}; next lowest {o.idxmin()} {o.min():.3f} (gap {col['cause'] - o.min():+.3f}); "
              f"others mean {o.mean():.3f}; below cause: {list(o[o < col['cause']].index)}")
    if sc == "raw":
        b = Q[Q.item == "cause"][["corpus", "n", "mean", "baseline_n", "baseline", "diff", "ci_low", "ci_high", "cohens_d", "p_holm"]]
        print("cause vs all other lexical verbs (corpus baseline):"); print(b.to_string(index=False))
