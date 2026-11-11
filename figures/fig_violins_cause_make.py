"""Violin plots of the effect sentiment of 'cause' and 'make' only (added 3 Nov 2026 at Dean's request).

fig_violins_cause_make_by_corpus_<scoring>   one violin per corpus and expression: in each corpus the 'cause' violin
                                             stands beside the 'make' violin (enTenTen21, BNC, COCA, AskReddit)
fig_violins_cause_make_pooled_<scoring>      the four corpora pooled: one violin for 'cause', one for 'make'
Same data and conventions as fig_violins.py: every line that enters the results (line score = the effect noun N, or the
mean of N and V when there is a complement; 'make' = causative frames only), kernel density with the same bandwidth,
all violins drawn to the same width, line = median, marker = mean.
Input: data/results/lines_<scoring>.pkl and pvalues_all.tsv (step 9). Output: data/figures/fig_violins_cause_make_*.png/.pdf
Usage: python figures/fig_violins_cause_make.py [scoring]     (raw, band, adj or war; default raw)
"""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from causesent import config
import causesent.style  # Helvetica
from causesent.palette import bounds, band_colour_fn

sc = sys.argv[1] if len(sys.argv) > 1 else "raw"
config.ensure_dirs()
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
LO, HI = bounds(sc); col = band_colour_fn(LO, HI)      # grey band = lowest to highest of the four corpora's baselines
LABEL = {"raw": "raw scores", "band": "neutral band (4–6) removed", "adj": "adjectives and noun compounds averaged into the noun",
         "war": "Warriner et al. (2013) valence norms"}
SCALE = "Warriner valence" if sc == "war" else "Hedonometer score"
MK = {"ententen21": ("o", "enTenTen21", "#2b6cd4"), "bnc": ("s", "BNC", "#f2b705"),
      "coca": ("^", "COCA", "#1a9e8f"), "reddit": ("D", "AskReddit", "#8e3fb5")}
ORDER = ["ententen21", "bnc", "coca", "reddit"]
BASE = "BASELINE (other verbs)"
ITEMS = ["cause", "make"]
MIN_N = 30

def tidy(ax):
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8.5)

def violins(ax, data, pos, width, faces, edge=INK):
    parts = ax.violinplot(data, positions=pos, widths=width, showextrema=False, showmeans=False, showmedians=False,
                          points=120, bw_method=0.25)          # same bandwidth as fig_violins.py
    for b, fc in zip(parts["bodies"], faces):
        b.set_facecolor(fc); b.set_edgecolor(edge); b.set_linewidth(0.5); b.set_alpha(None); b.set_zorder(2)

L = pd.read_pickle(config.RESULTS / f"lines_{sc}.pkl")
L = L[L.agg_score.notna() & L.item.isin(ITEMS + [BASE])]
G = {k: g.agg_score.values for k, g in L.groupby(["corpus", "item"])}
P = pd.read_csv(config.RESULTS / "pvalues_all.tsv", sep="\t")
P = P[(P.scoring == sc) & (P.measure == "NV") & (P.polarity == "combined") & (P.baseline_type == "verbs")]
for (c, it), n in P[P.item.isin(ITEMS + [BASE])].set_index(["corpus", "item"]).n.items():   # same lines as the statistics
    if len(G.get((c, it), [])) != n: print(f"warning: {c} {it}: {len(G.get((c, it), []))} lines here, n = {n} in pvalues_all.tsv")
for c in ORDER:
    for it in ITEMS:
        assert len(G[(c, it)]) >= MIN_N, (c, it)

# ---------------------------------------------------------------- figure 1: by corpus
fig, ax = plt.subplots(figsize=(11.5, 6.4))
SP, OFF, W = 3.0, 0.62, 0.95
tr = ax.get_xaxis_transform()
for g, c in enumerate(ORDER):
    mk, name, fc = MK[c]
    for j, it in enumerate(ITEMS):
        x = SP * g + (-OFF if j == 0 else OFF)
        d = G[(c, it)]
        violins(ax, [d], [x], W, [matplotlib.colors.to_rgba(fc, 0.78 if j == 0 else 0.30)])
        ax.hlines(np.median(d), x - 0.2, x + 0.2, color=INK, lw=1.1, zorder=3)
        ax.scatter([x], [d.mean()], marker=mk, s=40, facecolor="white", edgecolor=INK, linewidths=0.9, zorder=4)
        ax.text(x, -0.035, it, transform=tr, ha="center", va="top", fontsize=11, fontstyle="italic", color=INK)
        ax.text(x, -0.095, f"mean {d.mean():.2f}", transform=tr, ha="center", va="top", fontsize=8.3, color=MUTED)
        ax.text(x, -0.14, f"n = {len(d):,}", transform=tr, ha="center", va="top", fontsize=8.3, color=MUTED)
    bl = G[(c, BASE)].mean()
    ax.hlines(bl, SP * g - 1.3, SP * g + 1.3, color=fc, lw=1.3, ls="--", zorder=1)
    ax.text(SP * g, -0.215, name, transform=tr, ha="center", va="top", fontsize=11.5, color=INK)
    ax.plot([SP * g - 1.25, SP * g + 1.25], [-0.2, -0.2], transform=tr, color=GRID, lw=1, clip_on=False)
ax.axhline(5, color=MUTED, lw=0.7, ls=":", zorder=1)
ax.set_xticks([]); ax.set_xlim(-1.5, SP * (len(ORDER) - 1) + 1.5)
ax.set_ylim(1, 9); ax.set_yticks(range(1, 10))
ax.set_ylabel(f"{SCALE} of the effect (per line)", fontsize=9.5, color=MUTED)
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0); tidy(ax)
h = [Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK, markersize=6),
     Line2D([], [], color=INK, lw=1.1), Line2D([], [], color=MUTED, lw=1.3, ls="--"), Line2D([], [], color=MUTED, lw=0.7, ls=":")]
ax.legend(h, ["mean (marker shape = corpus)", "median", "mean of all other verbs in that corpus (dashed, corpus colour)", "neutral (5)"],
          loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8.3, ncol=2, frameon=False)
fig.suptitle(f"Effect sentiment of cause and make, per line, in each corpus ({LABEL[sc]})", fontsize=12, x=0.065, ha="left", y=0.985)
fig.text(0.01, 0.012, "Each violin = the distribution of the line scores of one expression in one corpus (kernel density, all violins drawn to the same width, so the width shows the shape, "
         "not the number of lines).\nDark = cause, light = make. Line score = the effect noun N, or the mean of N and V when there is a complement. make: causative frames only (make N ADJ / N V / N V-ed).",
         fontsize=7.6, color=MUTED, va="bottom")
fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.27)
out = str(config.FIGURES / f"fig_violins_cause_make_by_corpus_{sc}.png"); fig.savefig(out, dpi=200); fig.savefig(out.replace(".png", ".pdf")); plt.close(fig)
print("saved", out)

# ---------------------------------------------------------------- figure 2: the four corpora pooled
pooled = {it: np.concatenate([G[(c, it)] for c in ORDER]) for it in ITEMS + [BASE]}
fig, ax = plt.subplots(figsize=(6.6, 6.2))
tr = ax.get_xaxis_transform()
for j, it in enumerate(ITEMS):
    d = pooled[it]
    violins(ax, [d], [j], 0.8, [col(d.mean())])
    ax.hlines(np.median(d), j - 0.25, j + 0.25, color=INK, lw=1.2, zorder=3)
    ax.scatter([j], [d.mean()], marker="o", s=48, facecolor="white", edgecolor=INK, linewidths=0.9, zorder=4)
    ax.text(j, -0.035, it, transform=tr, ha="center", va="top", fontsize=12, fontstyle="italic", color=INK)
    ax.text(j, -0.1, f"mean {d.mean():.2f}, median {np.median(d):.2f}", transform=tr, ha="center", va="top", fontsize=8.6, color=MUTED)
    ax.text(j, -0.145, f"n = {len(d):,}", transform=tr, ha="center", va="top", fontsize=8.6, color=MUTED)
bl = pooled[BASE].mean()
ax.axhline(bl, color=MUTED, lw=1.2, ls="--", zorder=1)
ax.axhline(5, color=MUTED, lw=0.7, ls=":", zorder=1)
ax.set_xticks([]); ax.set_xlim(-0.75, 1.75); ax.set_ylim(1, 9); ax.set_yticks(range(1, 10))
ax.set_ylabel(f"{SCALE} of the effect (per line)", fontsize=9.5, color=MUTED)
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0); tidy(ax)
h = [Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK, markersize=6), Line2D([], [], color=INK, lw=1.2),
     Line2D([], [], color=MUTED, lw=1.2, ls="--"), Line2D([], [], color=MUTED, lw=0.7, ls=":")]
ax.legend(h, ["mean", "median", f"all other verbs, pooled ({bl:.2f})", "neutral (5)"], loc="lower left", bbox_to_anchor=(0, 1.0),
          fontsize=8.3, ncol=2, frameon=False)
fig.suptitle(f"Effect sentiment of cause and make, per line, four corpora pooled ({LABEL[sc]})", fontsize=11.5, x=0.1, ha="left", y=0.985)
share = lambda it: ", ".join(f"{MK[c][1]} {100 * len(G[(c, it)]) / len(pooled[it]):.0f}%" for c in ORDER)
fig.text(0.01, 0.012, "Lines of enTenTen21, BNC, COCA and AskReddit pooled, each line counting once, so the corpora enter in proportion to their number of lines.\n"
         f"Share of the lines: cause: {share('cause')};\nmake: {share('make')}. Violin colour = mean (grey = within the range of the corpus baselines).\n"
         "Violins drawn to the same width (shape, not sample size). make: causative frames only (make N ADJ / N V / N V-ed).",
         fontsize=7.4, color=MUTED, va="bottom")
fig.subplots_adjust(left=0.11, right=0.97, top=0.84, bottom=0.3)
out = str(config.FIGURES / f"fig_violins_cause_make_pooled_{sc}.png"); fig.savefig(out, dpi=200); fig.savefig(out.replace(".png", ".pdf")); plt.close(fig)
print("saved", out)
for it in ITEMS: print(it, "pooled n", len(pooled[it]), "mean %.3f median %.3f" % (pooled[it].mean(), np.median(pooled[it])), "| shares:", share(it))
print("pooled baseline mean %.3f" % bl)

# ---------------------------------------------------------------- figure 3: pooled + by corpus in ONE figure (3 Nov 2026, Dean)
fig, (axp, axc) = plt.subplots(1, 2, figsize=(15, 6.8), sharey=True, gridspec_kw={"width_ratios": [2.6, 12], "wspace": 0.04})
GREY = "#52514e"
for j, it in enumerate(ITEMS):                                   # left: the four corpora pooled
    d = pooled[it]; trp = axp.get_xaxis_transform()
    violins(axp, [d], [j], 0.85, [matplotlib.colors.to_rgba(GREY, 0.78 if j == 0 else 0.30)])
    axp.hlines(np.median(d), j - 0.22, j + 0.22, color=INK, lw=1.1, zorder=3)
    axp.scatter([j], [d.mean()], marker="o", s=40, facecolor="white", edgecolor=INK, linewidths=0.9, zorder=4)
    axp.text(j, -0.035, it, transform=trp, ha="center", va="top", fontsize=11, fontstyle="italic", color=INK)
    axp.text(j, -0.095, f"mean {d.mean():.2f}", transform=trp, ha="center", va="top", fontsize=8.3, color=MUTED)
    axp.text(j, -0.14, f"n = {len(d):,}", transform=trp, ha="center", va="top", fontsize=8.3, color=MUTED)
axp.axhline(pooled[BASE].mean(), color=MUTED, lw=1.3, ls="--", zorder=1)
axp.text(0.5, -0.215, "all four corpora pooled", transform=axp.get_xaxis_transform(), ha="center", va="top", fontsize=11.5, color=INK)
axp.plot([-0.55, 1.55], [-0.2, -0.2], transform=axp.get_xaxis_transform(), color=GRID, lw=1, clip_on=False)
axp.set_xlim(-0.65, 1.65)
trc = axc.get_xaxis_transform()
for g, c in enumerate(ORDER):                                    # right: one pair per corpus
    mk, name, fc = MK[c]
    for j, it in enumerate(ITEMS):
        x = SP * g + (-OFF if j == 0 else OFF); d = G[(c, it)]
        violins(axc, [d], [x], W, [matplotlib.colors.to_rgba(fc, 0.78 if j == 0 else 0.30)])
        axc.hlines(np.median(d), x - 0.2, x + 0.2, color=INK, lw=1.1, zorder=3)
        axc.scatter([x], [d.mean()], marker=mk, s=40, facecolor="white", edgecolor=INK, linewidths=0.9, zorder=4)
        axc.text(x, -0.035, it, transform=trc, ha="center", va="top", fontsize=11, fontstyle="italic", color=INK)
        axc.text(x, -0.095, f"mean {d.mean():.2f}", transform=trc, ha="center", va="top", fontsize=8.3, color=MUTED)
        axc.text(x, -0.14, f"n = {len(d):,}", transform=trc, ha="center", va="top", fontsize=8.3, color=MUTED)
    axc.hlines(G[(c, BASE)].mean(), SP * g - 1.3, SP * g + 1.3, color=fc, lw=1.3, ls="--", zorder=1)
    axc.text(SP * g, -0.215, name, transform=trc, ha="center", va="top", fontsize=11.5, color=INK)
    axc.plot([SP * g - 1.25, SP * g + 1.25], [-0.2, -0.2], transform=trc, color=GRID, lw=1, clip_on=False)
axc.set_xlim(-1.5, SP * (len(ORDER) - 1) + 1.5)
for ax in (axp, axc):
    ax.axhline(5, color=MUTED, lw=0.7, ls=":", zorder=1)
    ax.set_xticks([]); ax.set_ylim(1, 9); ax.grid(axis="y", color=GRID, lw=0.6, zorder=0); tidy(ax)
axp.set_yticks(range(1, 10)); axp.set_ylabel(f"{SCALE} of the effect (per line)", fontsize=9.5, color=MUTED)
axc.tick_params(axis="y", left=False); axc.spines["left"].set_visible(False)
h = [Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK, markersize=6), Line2D([], [], color=INK, lw=1.1),
     Line2D([], [], color=MUTED, lw=1.3, ls="--"), Line2D([], [], color=MUTED, lw=0.7, ls=":")]
axp.legend(h, ["mean (marker shape = corpus; circle = pooled)", "median", "mean of all other verbs (pooled / in that corpus; corpus colour)", "neutral (5)"],
           loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8.3, ncol=4, frameon=False)
fig.suptitle(f"Effect sentiment of cause and make, per line: all four corpora pooled, and each corpus ({LABEL[sc]})", fontsize=12, x=0.065, ha="left", y=0.985)
fig.text(0.01, 0.012, "Each violin = the distribution of the line scores of one expression (kernel density, all violins drawn to the same width, so the width shows the shape, not the number of lines). Dark = cause, light = make.\n"
         f"Pooled = the lines of all four corpora together, each line counting once, so the corpora enter in proportion to their number of lines (share of cause: {share('cause')}; make: {share('make')}).\n"
         "Line score = the effect noun N, or the mean of N and V when there is a complement. make: causative frames only (make N ADJ / N V / N V-ed).",
         fontsize=7.6, color=MUTED, va="bottom")
fig.subplots_adjust(left=0.06, right=0.995, top=0.83, bottom=0.27)
out = str(config.FIGURES / f"fig_violins_cause_make_pooled_and_by_corpus_{sc}.png"); fig.savefig(out, dpi=200); fig.savefig(out.replace(".png", ".pdf")); plt.close(fig)
print("saved", out)
