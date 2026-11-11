"""Violin version of fig_effect_sentiment_cause_allow_make_force (4 Nov 2026, Dean): cause, allow, make, force.

One violin per expression instead of one bar: the distribution of the line scores with each of the four corpora weighted 25%
(a weighted kernel density: every line of corpus c has weight 0.25 / n_c, so the violin is not dominated by AskReddit and its mean is
the same 'mean of the four corpora's means' as the bar was). Black dash = that mean; one marker per corpus at the corpus's own mean
(filled = significantly different from that corpus's baseline of all other verbs, Welch t-test, Holm; hollow = not); dashed lines = baselines.
Input: data/results/lines_<scoring>.pkl and pvalues_all.tsv. Output: data/figures/fig_violins_cause_allow_make_force_<scoring>.png/.pdf
Usage: python figures/fig_violins_four_expressions.py [scoring]
"""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from causesent import config
import causesent.style  # Helvetica
from causesent.targets import DISPLAY
from causesent.palette import bounds, band_colour_fn

sc = sys.argv[1] if len(sys.argv) > 1 else "raw"
config.ensure_dirs(); out = str(config.FIGURES / f"fig_violins_cause_allow_make_force_{sc}.png")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
LO, HI = bounds(sc); col = band_colour_fn(LO, HI)
LABEL = {"raw": "raw scores", "band": "neutral band (4–6) removed", "adj": "adjectives and noun compounds averaged into the noun", "war": "Warriner et al. (2013) valence norms"}
SCALE = "Warriner valence" if sc == "war" else "Hedonometer score"
ITEMS = ["cause", "allow", "make", "force"]; MIN_N = 30
MK = {"ententen21": ("o", "enTenTen21 (web, 52B words)", "#2b6cd4"), "bnc": ("s", "BNC (British, 100M words)", "#f2b705"),
      "reddit": ("D", "AskReddit Jan 2018 (139M words)", "#8e3fb5"), "coca": ("^", "COCA (American, 1B words)", "#1a9e8f")}
off = {"bnc": -0.27, "ententen21": -0.09, "reddit": 0.09, "coca": 0.27}
lab = lambda it: DISPLAY.get(it, it) + (" *" if it in ("make", "get") else "")

P = pd.read_csv(config.RESULTS / "pvalues_all.tsv", sep="\t")
P = P[(P.scoring == sc) & (P.measure == "NV") & (P.polarity == "combined") & (P.baseline_type == "verbs")]
Q = P[(P.n >= MIN_N) & P.item.isin(ITEMS)]
W = Q.pivot(index="item", columns="corpus", values="mean"); S = Q.pivot(index="item", columns="corpus", values="p_holm")
mean4 = W.mean(axis=1).sort_values(); items = list(mean4.index); x = np.arange(len(items))
L = pd.read_pickle(config.RESULTS / f"lines_{sc}.pkl"); L = L[L.agg_score.notna() & L.item.isin(ITEMS)]
G = {k: g.agg_score.values for k, g in L.groupby(["corpus", "item"])}

fig, ax = plt.subplots(figsize=(9.5, 7.4))
HALF = 0.40
for xi, it in zip(x, items):
    cs = [c for c in MK if len(G.get((c, it), [])) >= MIN_N]
    data = np.concatenate([G[(c, it)] for c in cs]); w = np.concatenate([np.full(len(G[(c, it)]), 1 / len(cs) / len(G[(c, it)])) for c in cs])
    wmean = np.average(data, weights=w); assert abs(wmean - mean4[it]) < 1e-4, (it, wmean, mean4[it])      # = the bar of the bar chart
    kde = gaussian_kde(data, weights=w, bw_method=0.25)
    g = np.linspace(data.min(), data.max(), 300); d = kde(g); d = d / d.max() * HALF
    ax.fill_betweenx(g, xi - d, xi + d, facecolor=col(wmean), edgecolor=INK, linewidth=0.5, zorder=2)
    ax.hlines(wmean, xi - HALF * 0.9, xi + HALF * 0.9, color=INK, lw=1.6, zorder=3)
for c, (mk, name, fc) in MK.items():
    y = W[c].reindex(items).values; sig = (S[c].reindex(items) < 0.05).values
    ax.scatter((x + off[c])[sig], y[sig], marker=mk, s=46, facecolor=fc, edgecolor=INK, linewidths=0.6, zorder=4, label=name)
    ax.scatter((x + off[c])[~sig], y[~sig], marker=mk, s=46, facecolor="white", edgecolor=fc, linewidths=1.3, zorder=4)
    bl = P[P.corpus == c].baseline.dropna()
    if len(bl): ax.axhline(bl.iloc[0], color=fc, lw=1.0, ls="--", zorder=1, label=f"{name.split(' (')[0]} baseline, all other verbs: {bl.iloc[0]:.2f}")
ax.axhline(5, color=MUTED, lw=0.7, ls=":", zorder=0)
ax.set_xticks(x); ax.set_xticklabels([lab(i) for i in items], fontsize=12, fontstyle="italic")
ax.set_ylim(1, 10.7); ax.set_yticks(range(1, 10)); ax.set_xlim(-0.6, len(items) - 0.4)
ax.set_ylabel(f"{SCALE} of the effect (per line)", fontsize=8.5, color=MUTED)
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.tick_params(colors=MUTED, labelsize=8)
h, l = ax.get_legend_handles_labels()
h.append(Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor=MUTED)); l.append("hollow marker = not significantly different from that corpus's baseline (Holm)")
h.append(Line2D([], [], color=INK, lw=1.6)); l.append("mean of the four corpora (each 25%)")
ax.legend(h, l, loc="upper left", fontsize=7.4, frameon=True, framealpha=0.95, edgecolor="none", ncol=2)
ax.set_title(f"Effect sentiment of cause, allow, make and force, per line ({LABEL[sc]}; violin = four corpora weighted equally, markers = each corpus)", fontsize=10, loc="left")
import textwrap
foot = (f"Violin = distribution of the line scores, each corpus weighted 25% (weighted kernel density, all violins drawn to the same width). "
        f"Colour: grey = between the lowest and highest of the four corpora's baselines ({LO:.2f}–{HI:.2f}), fading to red below and to green above; dotted line = 5. "
        "Filled marker = significantly different from that corpus's baseline (Welch t-test, Holm-corrected, p < .05). Same order and means as the bar chart. "
        "Baselines = every other lexical verb, same rules as in the 101-expression figure. * make: causative frames only (make N ADJ / N V / N V-ed).")
fig.text(0.01, 0.01, textwrap.fill(foot, 165), fontsize=7, color=MUTED, va="bottom")
fig.subplots_adjust(left=0.1, right=0.99, top=0.93, bottom=0.17)
fig.savefig(out, dpi=200); fig.savefig(out.replace(".png", ".pdf")); print("saved", out)
