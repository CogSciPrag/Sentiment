"""ONLY-N / ONLY-V version of fig_violins_sufficiency_groups.py (5 Nov 2026, Dean): the same 20 expressions in the same three groups,
but each line is scored on the effect NOUN only (measure N, column score_N) or the effect VERB only (measure V, column score_V),
as in the earlier cause / force / make / let only-N and only-V violins. Baselines, significance and the grey band come from the
same measure in pvalues_all.tsv. Cells (expression x corpus) with < 30 scored lines are left out: no marker, and the violin and
its mean use the remaining corpora (equal weights); the tick label then says how many corpora are shown.
Usage: python figures/fig_violins_sufficiency_groups_N_V.py N|V [scoring]

Original docstring:
Violin plot of 18 expressions in two groups (4 Nov 2026, Dean): clearly sufficiency-entailing (9) vs clearly not sufficiency-entailing (9).

One violin per expression instead of one bar: the distribution of the line scores with each of the four corpora weighted 25%
(a weighted kernel density: every line of corpus c has weight 0.25 / n_c, so the violin is not dominated by AskReddit and its mean is
the same 'mean of the four corpora's means' as the bar was). Black dash = that mean; one marker per corpus at the corpus's own mean
(filled = significantly different from that corpus's baseline of all other verbs, Welch t-test, Holm; hollow = not); dashed lines = baselines.
Input: data/results/lines_<scoring>.pkl and pvalues_all.tsv. Output: data/figures/fig_violins_sufficiency_groups_<scoring>.png/.pdf + .tsv
Usage: python figures/fig_violins_sufficiency_groups.py [scoring]
"""
import sys
import numpy as np, pandas as pd, matplotlib, matplotlib.transforms
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from causesent import config
import causesent.style  # Helvetica
from causesent.targets import DISPLAY
from causesent.palette import bounds, band_colour_fn

M = sys.argv[1] if len(sys.argv) > 1 else "N"; assert M in ("N", "V")
sc = sys.argv[2] if len(sys.argv) > 2 else "raw"
COL = {"N": "score_N", "V": "score_V"}[M]
ONLY = {"N": "only the effect noun ('only N' frame)", "V": "only the effect verb ('only V' frame)"}[M]
config.ensure_dirs(); out = str(config.FIGURES / f"fig_violins_sufficiency_groups_only{M}_{sc}.png")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
LO, HI = bounds(sc, M); col = band_colour_fn(LO, HI)
LABEL = {"raw": "raw scores", "band": "neutral band (4–6) removed", "adj": "adjectives and noun compounds averaged into the noun", "war": "Warriner et al. (2013) valence norms"}
SCALE = "Warriner valence" if sc == "war" else "Hedonometer score"
GROUPS = {"Cause and because": ["cause", "because (clause)"],
          "Clearly sufficiency-entailing": ["compel", "enough for", "ensure", "force", "guarantee", "make", "oblige", "produce", "sufficient for"],
          "Clearly not sufficiency-entailing": ["allow", "contribute to", "enable", "encourage", "influence", "let", "make a difference to", "necessary for", "permit"]}
ITEMS = [i for g in GROUPS.values() for i in g]; MIN_N = 30
MK = {"ententen21": ("o", "enTenTen21 (web, 52B words)", "#2b6cd4"), "bnc": ("s", "BNC (British, 100M words)", "#f2b705"),
      "reddit": ("D", "AskReddit Jan 2018 (139M words)", "#8e3fb5"), "coca": ("^", "COCA (American, 1B words)", "#1a9e8f")}
off = {"bnc": -0.28, "ententen21": -0.095, "reddit": 0.095, "coca": 0.28}
DISPLAY = {**DISPLAY, "because (clause)": "because"}
lab = lambda it: DISPLAY.get(it, it)   # no asterisk on make (Dean, 4 Nov 2026)

P = pd.read_csv(config.RESULTS / "pvalues_all.tsv", sep="\t")
P = P[(P.scoring == sc) & (P.measure == M) & (P.polarity == "combined") & (P.baseline_type == "verbs")]
Q = P[(P.n >= MIN_N) & P.item.isin(ITEMS)]
W = Q.pivot(index="item", columns="corpus", values="mean"); S = Q.pivot(index="item", columns="corpus", values="p_holm")
mean4 = W.mean(axis=1)
items = []; x = []; gx = {}; pos = 0.0
for gname, its in GROUPS.items():
    its = sorted(its, key=lambda i: mean4[i]); start = pos
    for i in its: items.append(i); x.append(pos); pos += 1
    gx[gname] = (start, pos - 1); pos += 0.9
x = np.array(x)
L = pd.read_pickle(config.RESULTS / f"lines_{sc}.pkl"); L = L[L[COL].notna() & L.item.isin(ITEMS)]
G = {k: g[COL].values for k, g in L.groupby(["corpus", "item"])}

fig, ax = plt.subplots(figsize=(24, 8.5))   # wider for the larger labels (5 Nov 2026, Dean)
HALF = 0.42
NC = {}
for xi, it in zip(x, items):
    cs = [c for c in MK if len(G.get((c, it), [])) >= MIN_N]; assert len(cs) >= 1, (it, cs); NC[it] = len(cs)
    data = np.concatenate([G[(c, it)] for c in cs]); w = np.concatenate([np.full(len(G[(c, it)]), 1 / len(cs) / len(G[(c, it)])) for c in cs])
    wmean = np.average(data, weights=w); assert abs(wmean - mean4[it]) < 1e-4, (it, wmean, mean4[it])      # = the bar of the bar chart
    kde = gaussian_kde(data, weights=w, bw_method=0.25)
    g = np.linspace(data.min(), data.max(), 300); d = kde(g); d = d / d.max() * HALF
    ax.fill_betweenx(g, xi - d, xi + d, facecolor=col(wmean), edgecolor=INK, linewidth=0.5, zorder=2)
    ax.hlines(wmean, xi - HALF * 0.9, xi + HALF * 0.9, color=INK, lw=1.6, zorder=3)
for c, (mk, name, fc) in MK.items():
    y = W[c].reindex(items).values; ok = ~np.isnan(y); sig = ((S[c].reindex(items) < 0.05).values) & ok; ns = ok & ~sig
    ax.scatter((x + off[c])[sig], y[sig], marker=mk, s=30, facecolor=fc, edgecolor=INK, linewidths=0.6, zorder=4, label=name)
    ax.scatter((x + off[c])[ns], y[ns], marker=mk, s=30, facecolor="white", edgecolor=fc, linewidths=1.3, zorder=4)
    med = np.array([np.median(G[(c, i)]) if len(G.get((c, i), [])) >= MIN_N else np.nan for i in items])
    ax.hlines(med[ok], (x + off[c] - 0.08)[ok], (x + off[c] + 0.08)[ok], color=fc, lw=2.2, zorder=3)
    bl = P[P.corpus == c].baseline.dropna()
    if len(bl): ax.axhline(bl.iloc[0], color=fc, lw=1.0, ls="--", zorder=1, label=f"{name.split(' (')[0]} baseline, all other verbs: {bl.iloc[0]:.2f}")
ax.axhline(5, color=MUTED, lw=0.7, ls=":", zorder=0)
BREAK = {"sufficient for": "sufficient\nfor", "enough for": "enough\nfor", "contribute to": "contribute\nto", "necessary for": "necessary\nfor", "make a difference to": "make a\ndifference\nto"}   # hand-set line breaks (no mid-word splits)
ax.set_xticks(x); ax.set_xticklabels([BREAK.get(i, lab(i)) for i in items], fontsize=14.5, fontstyle="italic")
for xi, it in zip(x, items):     # fewer than four corpora with >= 30 lines: say so under the violin
    if NC[it] < 4: ax.text(xi, -0.2, f"{NC[it]} of 4 corpora\nwith n ≥ 30", ha="center", va="top", fontsize=10, color=MUTED, transform=matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes))
for gname, (a, b) in gx.items():
    ax.plot([a - 0.4, b + 0.4], [9.25, 9.25], color=MUTED, lw=1.0)          # group labels above the violins (4 Nov 2026, Dean)
    ax.text((a + b) / 2, 9.35, gname, ha="center", va="bottom", fontsize=11.5, color=INK, fontweight="normal")
_g = list(gx.values())
for (_, e1), (s2, _) in zip(_g[:-1], _g[1:]): ax.vlines((e1 + s2) / 2, 1, 9, color=MUTED, lw=0.8, ls=(0, (2, 3)), zorder=0)   # dashed dividers stop at 9
ax.set_ylim(1, 10.0); ax.set_yticks(range(1, 10)); ax.set_xlim(-0.65, x.max() + 0.65)
ax.set_ylabel(f"{SCALE} of the effect (per line)\n{ONLY}", fontsize=13, color=INK)   # larger axis labels (5 Nov 2026, Dean)
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.tick_params(colors=MUTED, labelsize=13); ax.tick_params(axis="x", labelsize=14.5, labelcolor=INK)   # word labels larger (4 Nov 2026, Dean)
h, l = ax.get_legend_handles_labels()
h.append(Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor=MUTED)); l.append("hollow marker = not significantly different from that corpus's baseline (Holm)")
h.append(Line2D([], [], color=INK, lw=1.6)); l.append("mean of the four corpora (each 25%)")
h.append(Line2D([], [], color=MUTED, lw=2.2)); l.append("short bar in the corpus colour = that corpus's median")
fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.055, 0.935), fontsize=7.6, frameon=False, ncol=3, columnspacing=2.0)
fig.text(0.06, 0.975, f"Effect sentiment of 20 expressions by group, per line ({LABEL[sc]}; {ONLY}; violin = four corpora weighted equally, markers = each corpus)", fontsize=11, ha="left", va="top")
import textwrap
foot = (f"Violin = distribution of the line scores, each corpus weighted 25% (weighted kernel density, all violins drawn to the same width). "
        f"Colour: grey = between the lowest and highest of the four corpora's baselines ({LO:.2f}–{HI:.2f}), fading to red below and to green above; dotted line = 5. "
        "Filled marker = significantly different from that corpus's baseline (Welch t-test, Holm-corrected, p < .05). Within each group the violins are ordered by their mean. All lines of an expression (all frames) are pooled; the n of every cell is in the .tsv. "
        "Baselines = every other lexical verb, same rules as in the 101-expression figure. * make: causative frames only (make N ADJ / N V / N V-ed). because = 'because (clause)' (the effect is the main clause, to the left of the word; 'because of' is a separate expression and is not shown).")
# footnote removed (Dean, 4 Nov 2026); text kept above for reference only
fig.subplots_adjust(left=0.06, right=0.995, top=0.855, bottom=0.2)
fig.savefig(out, dpi=200); fig.savefig(out.replace(".png", ".pdf")); Q.assign(group=Q.item.map({i: g for g, its in GROUPS.items() for i in its})).sort_values(["group", "item", "corpus"])[["group", "item", "corpus", "n", "mean", "baseline", "diff", "p_holm", "stars"]].to_csv(out.replace(".png", ".tsv"), sep="\t", index=False, float_format="%.5g")
print("saved", out); print(mean4.loc[items].round(3).to_string())
