"""Distributions of line scores, not just means: violin plots (added 2 Nov 2026 at Dean's request).

fig_violins_<scoring>      one panel per corpus; one violin per expression (all lines that enter the results, line
                           score = N, or the mean of N and V), in the order of the main figure (mean over the corpora);
                           violin colour = that corpus's mean on the red-grey-green scale centred at 5; line = median,
                           corpus marker = mean; the corpus's baseline of all other verbs as the first violin (grey) and
                           as a dashed line. Only expressions with >= 30 scored lines in that corpus.
fig_violins_key_<scoring>  the key expressions (cause, make, ensure, guarantee, secure, make sure) and the verb baseline, the four
                           corpora side by side (corpus colours and markers as in the other figures).
Input: data/results/lines_<scoring>.pkl and pvalues_all.tsv (step 9). Output: data/figures/fig_violins*_<scoring>.png/.pdf
Usage: python figures/fig_violins.py [scoring]     (raw, band, adj or war; default raw)
"""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from causesent import config
import causesent.style  # Helvetica
from causesent.targets import PREVENTION, EFFECT_LEFT, SYMMETRIC, DISPLAY

sc = sys.argv[1] if len(sys.argv) > 1 else "raw"
config.ensure_dirs()
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
CMAP = LinearSegmentedColormap.from_list("rg", ["#b2182b", "#e8816b", "#dcdad4", "#7fbf7b", "#1b7837"])
from causesent.palette import centre, bounds, band_colour_fn
CENTRE = centre(sc); LO, HI = bounds(sc); col = band_colour_fn(LO, HI)   # grey band = from the lowest to the highest of the four corpora's baselines (3 Nov 2026)
MIN_N = 30
LABEL = {"raw": "raw scores", "band": "neutral band (4–6) removed", "adj": "adjectives and noun compounds averaged into the noun",
         "war": "Warriner et al. (2013) valence norms"}
SCALE = "Warriner valence" if sc == "war" else "Hedonometer score"
YLIM = (1, 9)
MK = {"ententen21": ("o", "enTenTen21", "#2b6cd4"), "bnc": ("s", "BNC", "#f2b705"),
      "coca": ("^", "COCA", "#1a9e8f"), "reddit": ("D", "AskReddit", "#8e3fb5")}
ORDER = ["ententen21", "bnc", "coca", "reddit"]
BASE = "BASELINE (other verbs)"
KEY_ITEMS = ["cause", "make", "ensure", "guarantee", "secure", "make sure"]   # make added (Dean, 2 Nov 2026)
KEY = {"cause", "make", "ensure", "guarantee", "secure", "prevent", "trigger", "foster", "associated with", "correlated with"}
lab = lambda it: DISPLAY.get(it, it) + (" *" if it in ("make", "get") else "") + (" †" if it in PREVENTION else "") + (" ←" if it in EFFECT_LEFT else "") + (" ≈" if it in SYMMETRIC else "")

def tidy(ax):
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)

def violins(ax, data, pos, width, faces, edge=INK):
    """Violins scaled to the same width; returns nothing. data: list of arrays (len >= 2)."""
    parts = ax.violinplot(data, positions=pos, widths=width, showextrema=False, showmeans=False, showmedians=False,
                          points=120, bw_method=0.25)
    for b, fc in zip(parts["bodies"], faces):
        b.set_facecolor(fc); b.set_edgecolor(edge); b.set_linewidth(0.35); b.set_alpha(1); b.set_zorder(2)

L = pd.read_pickle(config.RESULTS / f"lines_{sc}.pkl")
L = L[L.agg_score.notna()]
P = pd.read_csv(config.RESULTS / "pvalues_all.tsv", sep="\t")
P = P[(P.scoring == sc) & (P.measure == "NV") & (P.polarity == "combined") & (P.baseline_type == "verbs")]
Q = P[P.n >= MIN_N]
W = Q.pivot(index="item", columns="corpus", values="mean")
order = list(W.mean(axis=1).sort_values().index)                 # as in the main figure
G = {k: g.agg_score.values for k, g in L.groupby(["corpus", "item"])}

# consistency check with the statistics: the same number of lines
for (c, it), n in Q.set_index(["corpus", "item"]).n.items():
    if len(G.get((c, it), [])) != n: print(f"warning: {c} {it}: {len(G.get((c, it), []))} lines here, n = {n} in pvalues_all.tsv")

# ---- figure 1: one panel per corpus, all expressions
fig, axes = plt.subplots(4, 1, figsize=(19, 22))
x = np.arange(len(order)) + 1.5        # position 0 = the baseline
for ax, c in zip(axes, ORDER):
    mk, name, fc = MK[c]
    have = [i for i, it in enumerate(order) if it in W.index and pd.notna(W.loc[it].get(c)) and len(G.get((c, it), [])) >= MIN_N]
    data = [G[(c, order[i])] for i in have]
    violins(ax, [G[(c, BASE)]], [0], 0.8, ["#cfcdc7"])
    violins(ax, data, x[have], 0.8, [col(np.mean(d)) for d in data])
    allpos = np.r_[0, x[have]]; alldata = [G[(c, BASE)]] + data
    med = [np.median(d) for d in alldata]; mean = [np.mean(d) for d in alldata]
    ax.hlines(med, allpos - 0.28, allpos + 0.28, color=INK, lw=0.9, zorder=3)
    ax.scatter(allpos, mean, marker=mk, s=16, facecolor=fc, edgecolor=INK, linewidths=0.5, zorder=4)
    for i, it in enumerate(order):
        if i not in have: ax.text(x[i], 1.15, "n<30" if it in W.index or True else "", rotation=90, fontsize=6, color=MUTED, ha="center", va="bottom")
    bl = np.mean(G[(c, BASE)])
    ax.axhline(bl, color=fc, lw=1.0, ls="--", zorder=1)
    ax.axhline(5, color=MUTED, lw=0.7, ls=":", zorder=1)
    ax.set_xticks(np.r_[0, x]); ax.set_xticklabels(["other verbs (baseline)"] + [lab(i) for i in order], rotation=90, fontsize=7.6, fontstyle="italic")
    ax.get_xticklabels()[0].set_fontstyle("normal")
    ax.set_xlim(-0.8, x[-1] + 0.8); ax.set_ylim(*YLIM); ax.set_yticks(range(1, 10))
    ax.set_ylabel(f"{SCALE} of the effect\n(per line)", fontsize=8.5, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0); tidy(ax)
    n_lines = sum(len(d) for d in data)
    ax.set_title(f"{name}: {len(have)} expressions, {n_lines:,} scored lines; baseline of all other verbs {bl:.2f} "
                 f"({len(G[(c, BASE)]):,} lines)", fontsize=10.5, loc="left")
    h = [Line2D([], [], marker=mk, ls="", markerfacecolor=fc, markeredgecolor=INK, markersize=5),
         Line2D([], [], color=INK, lw=0.9), Line2D([], [], color=fc, lw=1.0, ls="--"), Line2D([], [], color=MUTED, lw=0.7, ls=":")]
    ax.legend(h, ["mean", "median", f"baseline mean ({bl:.2f})", "neutral (5)"], loc="lower right", fontsize=7.5, ncol=4,
              frameon=True, framealpha=0.95, edgecolor="none")
fig.suptitle(f"Distribution of effect sentiment per expression and corpus ({LABEL[sc]}; ordered by the mean over the corpora, as in the main figure)",
             fontsize=13, x=0.01, ha="left", y=0.995)
fig.text(0.01, 0.004,
         "Each violin = the distribution of the line scores of one expression in one corpus (kernel density, all violins drawn to the same width, so the width shows\n"
         "the shape, not the number of lines). Line score = the effect noun N, or the mean of N and V when there is a complement. Violin colour = the corpus mean "
         "(grey = between the lowest and highest of the four corpora's baselines, fading to red below and to green above).\nOnly expressions with ≥ 30 scored lines in that corpus ('n<30' otherwise). COCA samples are small "
         "(about 275 scored lines per expression; thousands elsewhere), so its shapes are rougher.\n"
         "* make and get: causative frames only.   † prevention words and 'fail to': the complement is what does NOT happen.   ← effect on the left.   ≈ correlation: X as N, Y as V.",
         fontsize=7.4, color=MUTED, va="bottom")
fig.subplots_adjust(left=0.05, right=0.995, top=0.965, bottom=0.1, hspace=0.75)
out = str(config.FIGURES / f"fig_violins_{sc}.png"); fig.savefig(out, dpi=200); fig.savefig(out.replace(".png", ".pdf")); plt.close(fig)
print("saved", out)

# ---- figure 2: key expressions, corpora side by side
items = [BASE] + KEY_ITEMS
fig, ax = plt.subplots(figsize=(12, 5.6))
off = {"ententen21": -0.3, "bnc": -0.1, "coca": 0.1, "reddit": 0.3}
for c in ORDER:
    mk, name, fc = MK[c]
    pos, data = [], []
    for j, it in enumerate(items):
        d = G.get((c, it), np.array([]))
        if len(d) >= MIN_N: pos.append(j + off[c]); data.append(d)
    violins(ax, data, pos, 0.18, [matplotlib.colors.to_rgba(fc, 0.55)] * len(data))
    ax.hlines([np.median(d) for d in data], np.array(pos) - 0.07, np.array(pos) + 0.07, color=INK, lw=0.9, zorder=3)
    ax.scatter(pos, [np.mean(d) for d in data], marker=mk, s=24, facecolor="white", edgecolor=INK, linewidths=0.8, zorder=4)
    ax.fill_between([], [], color=fc, label=name)
ax.axhline(5, color=MUTED, lw=0.7, ls=":", zorder=1)
ax.set_xticks(range(len(items))); ax.set_xticklabels(["other verbs\n(baseline)"] + [lab(i) for i in KEY_ITEMS], fontsize=10)
for t in ax.get_xticklabels()[1:]: t.set_fontstyle("italic")
ax.set_ylim(*YLIM); ax.set_yticks(range(1, 10)); ax.set_xlim(-0.6, len(items) - 0.4)
ax.set_ylabel(f"{SCALE} of the effect (per line)", fontsize=9, color=MUTED)
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0); tidy(ax)
h, l = ax.get_legend_handles_labels()
h += [Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK, markersize=5), Line2D([], [], color=INK, lw=0.9),
      Line2D([], [], color=MUTED, lw=0.7, ls=":")]; l += ["mean (shape = corpus)", "median", "neutral (5)"]
ax.legend(h, l, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, ncol=7, frameon=False)
fig.suptitle(f"Effect sentiment of cause, make and the ensuring expressions, per line ({LABEL[sc]}); marker = mean", fontsize=10.5, x=0.07, ha="left", y=0.985)
fig.text(0.01, 0.01, "Violins drawn to the same width (shape, not sample size). All lines that enter the results; only with ≥ 30 scored lines. * make: causative frames only (make N ADJ / N V / N V-ed).",
         fontsize=7.4, color=MUTED, va="bottom")
fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.14)
out = str(config.FIGURES / f"fig_violins_key_{sc}.png"); fig.savefig(out, dpi=200); fig.savefig(out.replace(".png", ".pdf")); plt.close(fig)
print("saved", out)
