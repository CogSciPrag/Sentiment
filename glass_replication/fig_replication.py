"""The Glass replication: statistics for Glass's own measure on the whole month, and the replication figure.

1. Glass's measure (mean Hedonometer score of the verb's direct object) for each verb on the whole month of AskReddit,
   January 2018, against her baseline (all other verbs in the first 10 million words): Welch's t-test from the summary
   statistics in month_summary.tsv, Cohen's d, Holm correction -> data/results/glass/pvalues_glass_method.tsv
2. Figure: (A) replication check: Glass's posted output vs our re-run on her first 10 million words, and, for verbs too
   rare there, her unmodified code vs the fast re-run on the whole month; (B) her measure on the whole month.
   -> data/figures/fig_glass_replication.png and .pdf

Inputs (data/results/glass/): month_summary.tsv (summarise_month.py), validation_10M_glass_vs_ours.tsv (compare_val.py),
month_check.tsv (compare_month_check.py; optional), and month_summary_band.tsv for the 'band' version (optional).
Usage: python glass_replication/fig_replication.py [raw|band]
"""
import sys, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy import stats
from causesent import config
from causesent.stats import holm, stars

sc = sys.argv[1] if len(sys.argv) > 1 else "raw"
G = config.RESULTS / "glass"; config.ensure_dirs(); out = str(config.FIGURES / f"fig_glass_replication_{sc}.png")
MIN_N = config.MIN_N

# ---- 1. statistics for Glass's measure (from per-verb means, standard errors and counts)
rb = []
for scb, mfile in [("raw", "month_summary.tsv"), ("band", "month_summary_band.tsv")]:
    if not (G / mfile).exists(): continue
    M = pd.read_csv(G / mfile, sep="\t"); base = M[M.verb.str.startswith("BASELINE")].iloc[0]
    for _, r in M[~M.verb.str.startswith("BASELINE") & (M.dobj_n >= MIN_N)].iterrows():
        vx, vy = (r.dobj_se**2) * r.dobj_n, (base.dobj_se**2) * base.dobj_n
        se = np.sqrt(r.dobj_se**2 + base.dobj_se**2); d = r.dobj_mean - base.dobj_mean
        df = se**4 / (r.dobj_se**4 / (r.dobj_n - 1) + base.dobj_se**4 / (base.dobj_n - 1))
        p = 2 * stats.t.sf(abs(d / se), df)
        rb.append(dict(scoring=scb, verb=r.verb, n=int(r.dobj_n), mean=r.dobj_mean, baseline=base.dobj_mean, diff=d, p=p,
                       cohens_d=d / np.sqrt(((r.dobj_n - 1) * vx + (base.dobj_n - 1) * vy) / (r.dobj_n + base.dobj_n - 2))))
PB = pd.DataFrame(rb); PB["p_holm"] = np.nan
for _, idx in PB.groupby("scoring").groups.items(): PB.loc[idx, "p_holm"] = holm(PB.loc[idx, "p"].values)
PB["stars"] = PB.p_holm.map(stars)
PB.to_csv(G / "pvalues_glass_method.tsv", sep="\t", index=False, float_format="%.6g")
print(PB[PB.scoring == "raw"].sort_values("mean")[["verb", "n", "mean", "baseline", "cohens_d", "stars"]].round(3).head(8).to_string(index=False))

# ---- 2. figure
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
CMAP = LinearSegmentedColormap.from_list("rg", ["#b2182b", "#e8816b", "#dcdad4", "#7fbf7b", "#1b7837"])
NORM = TwoSlopeNorm(vmin=3.5, vcenter=5.0, vmax=6.5); col = lambda v: CMAP(NORM(min(max(v, 3.5), 6.5)))
MIN_N = 30
LABEL = {"raw": "raw scores", "band": "neutral band (4–6) removed", "adj": "adjectives and noun compounds averaged into the noun",
         "war": "Warriner et al. (2013) valence norms"}
MK = {"ententen21": ("o", "enTenTen21 (web, 52B words)", "#2b6cd4"), "bnc": ("s", "BNC (British, 100M words)", "#f2b705"),
      "reddit": ("D", "AskReddit Jan 2018 (139M words)", "#8e3fb5"), "coca": ("^", "COCA (American, 1B words)", "#1a9e8f")}
KEY = {"cause", "ensure", "guarantee", "secure", "prevent", "trigger", "foster", "associated with", "correlated with"}
def tidy(ax):
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)

PB = PB[PB.scoring == sc]
M = pd.read_csv(G / ("month_summary_band.tsv" if sc == "band" else "month_summary.tsv"), sep="\t")
V = pd.read_csv(G / "validation_10M_glass_vs_ours.tsv", sep="\t")
MC = pd.read_csv(G / "month_check.tsv", sep="\t") if (G / "month_check.tsv").exists() else None

fig = plt.figure(figsize=(17, 6.5))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 3.1], wspace=0.16)

# ---- A
ax = fig.add_subplot(gs[0])
v = V[(V.dobj_n_Glass >= MIN_N) & (V.verb != "ALL OTHER VERBS")]
ax.plot([3.4, 6.6], [3.4, 6.6], color=MUTED, lw=0.8, ls="--", zorder=1)
ax.scatter(v.dobj_mean_Glass, v.dobj_mean_ours, s=34, c=[col(x) for x in v.dobj_mean_ours], edgecolors=INK, linewidths=0.5, zorder=3,
           label=f"first 10M words: Glass's posted output vs our rerun ({len(v)} verbs)")
nm = 0
if MC is not None:
    mc = MC[(MC.dobj_n_Glass >= MIN_N) & (MC.dobj_n_ours >= MIN_N)]; nm = len(mc)
    ax.scatter(mc.dobj_mean_Glass, mc.dobj_mean_ours, s=40, marker="^", c=[col(x) for x in mc.dobj_mean_ours], edgecolors=INK, linewidths=0.5,
               zorder=3, label=f"whole month: Glass's code vs our pipeline ({nm} more verbs)")
    for _, r in mc.iterrows():
        if r.verb in ("ensure", "secure", "foster", "provoke", "induce", "compel", "facilitate"):
            ax.annotate(r.verb, (r.dobj_mean_Glass, r.dobj_mean_ours), xytext=(5, -3), textcoords="offset points", fontsize=7.5, fontstyle="italic")
for _, r in v.iterrows():
    if r.verb in ("cause", "prevent", "trigger", "guarantee"):
        ax.annotate(r.verb, (r.dobj_mean_Glass, r.dobj_mean_ours), xytext=(5, -3), textcoords="offset points", fontsize=7.5, fontstyle="italic")
ax.set_xlim(3.4, 6.6); ax.set_ylim(3.4, 6.6); ax.set_aspect("equal")
ax.set_xlabel("Glass's code: mean object score", fontsize=8.5, color=MUTED); ax.set_ylabel("Our pipeline: mean object score", fontsize=8.5, color=MUTED)
ax.grid(color=GRID, lw=0.6, zorder=0); tidy(ax)
ax.legend(loc="lower right", fontsize=6.4, frameon=True, framealpha=0.95, edgecolor="none")
ax.set_title("A. Replication check (raw Hedonometer, direct objects)", fontsize=10, loc="left")

# ---- B
ax = fig.add_subplot(gs[1])
b = M[~M.verb.str.startswith("BASELINE") & (M.dobj_n >= MIN_N)].sort_values("dobj_mean")
base = float(M[M.verb.str.startswith("BASELINE")].dobj_mean.iloc[0]); x = np.arange(len(b))
ax.bar(x, b.dobj_mean - 1, bottom=1, color=[col(m) for m in b.dobj_mean], edgecolor=INK, linewidth=0.4, width=0.75, zorder=2)
ax.errorbar(x, b.dobj_mean, yerr=1.96 * b.dobj_se, fmt="none", ecolor=INK, elinewidth=0.7, zorder=3)
st = PB.set_index("verb").stars
for xi, (_, r) in zip(x, b.iterrows()):
    s = st.get(r.verb, "")
    ax.text(xi, r.dobj_mean + 1.96 * r.dobj_se + 0.06, s, ha="center", va="bottom", fontsize=6.5 if s == "n.s." else 8, color=MUTED if s == "n.s." else INK, rotation=90 if s == "n.s." else 0)
ax.axhline(5, color=MUTED, lw=0.7, ls=":", zorder=1); ax.axhline(base, color=INK, lw=0.9, ls="--", zorder=1)
ax.text(-0.4, base + 0.04, f"all other verbs = {base:.2f}", ha="left", va="bottom", fontsize=7.8, color=INK)
ax.set_xticks(x); ax.set_xticklabels(b.verb, rotation=90, fontsize=8.3, fontstyle="italic")
for t in ax.get_xticklabels():
    if t.get_text() in KEY: t.set_fontweight("bold")
ax.set_ylim(1, 7.4); ax.set_xlim(-0.7, len(b) - 0.3)
ax.set_ylabel("Mean Hedonometer score\nof the direct object", fontsize=8.5, color=MUTED)
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0); tidy(ax)
ax.set_title("B. Glass's method, whole of AskReddit Jan 2018" + (" (neutral band removed)" if sc == "band" else ""),
             fontsize=10, loc="left")

fig.suptitle("Replicating Glass (2023) on AskReddit, January 2018", fontsize=14, x=0.01, ha="left", y=0.99)
fig.text(0.01, 0.01, "B: error bars = 95% CI; stars = difference from all other verbs (Welch t-test, Holm-corrected): *** p < .001, ** p < .01, * p < .05, n.s. = not significant. "
         "Only verbs with at least 30 scored objects.\nA: triangles = the whole month, where Glass's own per-sentence code (unmodified, spaCy model 3.6.0) and the fast re-run were run on the same sentences.",
         fontsize=7.2, color=MUTED, va="bottom")
fig.subplots_adjust(left=0.055, right=0.99, top=0.86, bottom=0.3)
fig.savefig(out, dpi=200); fig.savefig(out.replace(".png", ".pdf")); print("saved", out)
