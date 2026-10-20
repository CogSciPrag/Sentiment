"""Statistics: Welch's t-test with Cohen's d, Holm correction, and significance stars."""
import numpy as np, pandas as pd
from scipy import stats


def welch(x, y):
    """Welch's t-test of mean(x) - mean(y).
    Returns (difference, 95% CI low, 95% CI high, two-sided p, Cohen's d with the pooled standard deviation)."""
    nx, ny = len(x), len(y); vx, vy = x.var(ddof=1), y.var(ddof=1)
    d = x.mean() - y.mean(); se = np.sqrt(vx / nx + vy / ny)
    df = se**4 / ((vx / nx)**2 / (nx - 1) + (vy / ny)**2 / (ny - 1))
    t = stats.t.ppf(0.975, df); p = 2 * stats.t.sf(abs(d / se), df)
    sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    return d, d - t * se, d + t * se, p, d / sp


def holm(p):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    adj = np.minimum(1, np.maximum.accumulate(p[o] * (m - np.arange(m))))
    out = np.empty(m); out[o] = adj; return out


def stars(p):
    return "" if pd.isna(p) else "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
