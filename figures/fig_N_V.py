"""Figure: the effect sentiment split into its parts: N only (head noun), V only (complement verb / adjective /
predicate) and the two combined, per expression and corpus, each against that corpus's baseline on the same measure.
Input: data/results/pvalues_all.tsv (step 9). Output: data/figures/fig_N_V_NV_<scoring>.png and .pdf
Usage: python figures/fig_N_V.py [scoring]
"""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from causesent import config
from causesent.targets import PREVENTION, EFFECT_LEFT, SYMMETRIC

sc = sys.argv[1] if len(sys.argv) > 1 else "raw"
config.ensure_dirs(); out = str(config.FIGURES / f"fig_N_V_NV_{sc}.png")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
CMAP = LinearSegmentedColormap.from_list("rg", ["#b2182b", "#e8816b", "#dcdad4", "#7fbf7b", "#1b7837"])
NORM = TwoSlopeNorm(vmin=3.5, vcenter=5.0, vmax=6.5); col = lambda v: CMAP(NORM(min(max(v, 3.5), 6.5)))
LABEL = {"raw": "raw scores", "band": "neutral band (4–6) removed", "adj": "adjectives and noun compounds averaged into the noun",
         "war": "Warriner et al. (2013) valence norms"}
MK = {"ententen21": ("o", "enTenTen21", "#2b6cd4"), "bnc": ("s", "BNC", "#f2b705"), "reddit": ("D", "AskReddit", "#8e3fb5"), "coca": ("^", "COCA", "#1a9e8f")}
off = {"bnc": -0.27, "ententen21": -0.09, "reddit": 0.09, "coca": 0.27}
KEY = {"cause", "ensure", "guarantee", "secure", "prevent", "trigger", "foster", "associated with", "correlated with"}
lab = lambda it: it + (" *" if it in ("make", "get") else "") + (" †" if it in PREVENTION else "") + (" ←" if it in EFFECT_LEFT else "") + (" ≈" if it in SYMMETRIC else "")
TITLES = {"N": "1. Only N: head noun of the effect (or subject of a that-clause / left-side effect)" + (" + its adjectives and noun compounds" if sc == "adj" else "") + "; all lines",
          "V": "2. Only V: complement verb, adjective or predicate; only lines that have a complement",
          "NV": "3. N and V combined (as in figure C): N alone, or the mean of N and V when there is a complement"}
P = pd.read_csv(config.RESULTS / "pvalues_all.tsv", sep="\t"); P = P[P.scoring == sc]
NV = P[(P.measure == "NV") & (P.n >= 30)].pivot(index="item", columns="corpus", values="mean")
order = NV.mean(axis=1).sort_values().index.tolist()
fig, axes = plt.subplots(3, 1, figsize=(17, 20))
x = np.arange(len(order))
for ax, meas in zip(axes, ["N", "V", "NV"]):
    Q = P[(P.measure == meas) & (P.n >= 30)]
    W = Q.pivot(index="item", columns="corpus", values="mean").reindex(order)
    S = Q.pivot(index="item", columns="corpus", values="p_holm").reindex(order)
    bar = W.mean(axis=1); ok = bar.notna().values
    ax.bar(x[ok], bar.values[ok] - 1, bottom=1, color=[col(v) for v in bar.values[ok]], edgecolor=INK, linewidth=0.4, width=0.78, zorder=2)
    for c, (mk, name, fc) in MK.items():
        if c not in W: continue
        y = W[c].values; sig = (S[c] < 0.05).values
        ax.scatter((x + off[c])[sig], y[sig], marker=mk, s=20, facecolor=fc, edgecolor=INK, linewidths=0.6, zorder=4, label=name)
        ax.scatter((x + off[c])[~sig], y[~sig], marker=mk, s=20, facecolor="white", edgecolor=fc, linewidths=1.0, zorder=4)
        bl = P[(P.measure == meas) & (P.corpus == c)].baseline.dropna()
        if len(bl): ax.axhline(bl.iloc[0], color=fc, lw=1.0, ls="--", zorder=3, label=f"{name} baseline: {bl.iloc[0]:.2f}")
    for xi in x[~ok]: ax.text(xi, 1.15, "n<30", rotation=90, fontsize=6.5, color=MUTED, ha="center", va="bottom")
    ax.axhline(5, color=MUTED, lw=0.7, ls=":")
    ax.set_xticks(x); ax.set_xticklabels([lab(i) for i in order], rotation=90, fontsize=8.3, fontstyle="italic")
    for t, it in zip(ax.get_xticklabels(), order):
        if it in KEY: t.set_fontweight("bold")
    ax.set_ylim(1, 7.8); ax.set_yticks(range(1, 8)); ax.set_xlim(-0.8, len(order) - 0.2)
    ax.set_ylabel("Warriner valence" if sc == "war" else "Mean Hedonometer score", fontsize=8.5, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor=MUTED)); l.append("hollow = not significantly different from that corpus's baseline")
    ax.legend(h, l, loc="upper left", fontsize=7.6, frameon=True, framealpha=0.95, edgecolor="none", ncol=2)
    ax.set_title(TITLES[meas], fontsize=10.5, loc="left")
fig.suptitle(f"Effect sentiment of 78 causal expressions and 2 correlation expressions: N only, V only and combined ({LABEL[sc]})", fontsize=14, x=0.01, ha="left", y=0.99)
fig.text(0.01, 0.005, "Bars = mean of the corpora with ≥ 30 scored lines; markers = each corpus; dashed lines = each corpus's baseline of all other verbs on the same measure "
         "(Sketch Engine: 10,000 random verb lines; AskReddit: 3M-word sample; COCA: verbs in 5,500 random lines). Significance: Welch t-test vs that baseline, Holm-corrected within corpus and measure, p < .05.\n"
         "Bars start at 1; dotted line = neutral (5). Words ordered by panel 3 in all panels. Passives and negated effects excluded; with/for-N scored as the effect only for blame and credit.\n"
         "* make and get: causative frames only.   † prevention words and 'fail to': the complement is what does NOT happen.   ← effect is on the left.   ≈ correlation (non-causal comparison): X scored as N, Y as V.",
         fontsize=7.4, color=MUTED, va="bottom")
fig.subplots_adjust(left=0.05, right=0.99, top=0.955, bottom=0.1, hspace=0.6)
fig.savefig(out, dpi=200); fig.savefig(out.replace(".png", ".pdf")); print("ok")
