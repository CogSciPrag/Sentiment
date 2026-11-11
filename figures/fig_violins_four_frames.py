"""Violin plot of four specific frames (4 Nov 2026, Dean): cause N to V, make N V, force N to V, let N V.

Same design as the four-expression violin (fig_violins_four_expressions.py), but each violin only holds the lines of ONE frame:
  cause N to V  = item 'cause', frame 'V + N + to V'
  force N to V  = item 'force', frame 'V + N + to V'
  make N V      = item 'make',  frame 'V + N + bare V'
  let N V       = item 'let',   frame 'V + N + bare V'
Scores = per-line agg_score (noun and verb of the effect averaged), positive and negative forms combined.
Violin = weighted kernel density, each corpus weighted 25% (every line of corpus c has weight 0.25 / n_c);
black bar = mean of the four corpora's means (= weighted mean of the lines); marker = that corpus's own mean (filled = significantly
different from that corpus's baseline of all other verbs, Welch t-test, Holm over all 16 frame x corpus tests; hollow = not);
short bar in the corpus colour = that corpus's median.  Baseline = item 'BASELINE (other verbs)', the same lines as in the headline tables.
Input: data/results/lines_<scoring>.pkl.  Output: data/figures/fig_violins_four_frames_<scoring>.png/.pdf + a table (.tsv)
Usage: python figures/fig_violins_four_frames.py [scoring]
"""
import sys, textwrap
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from causesent import config
import causesent.style  # Helvetica
from causesent.stats import welch, holm
from causesent.palette import bounds, band_colour_fn

sc = sys.argv[1] if len(sys.argv) > 1 else "raw"
config.ensure_dirs(); out = str(config.FIGURES / f"fig_violins_four_frames_{sc}.png")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
LO, HI = bounds(sc); col = band_colour_fn(LO, HI)
LABEL = {"raw": "raw scores", "band": "neutral band (4–6) removed", "adj": "adjectives and noun compounds averaged into the noun", "war": "Warriner et al. (2013) valence norms"}
SCALE = "Warriner valence" if sc == "war" else "Hedonometer score"
FRAMES = {"cause N to V": ("cause", "V + N + to V"), "force N to V": ("force", "V + N + to V"),
          "make N V": ("make", "V + N + bare V"), "let N V": ("let", "V + N + bare V")}
MIN_N = 30; BASE = "BASELINE (other verbs)"
MK = {"ententen21": ("o", "enTenTen21 (web, 52B words)", "#2b6cd4"), "bnc": ("s", "BNC (British, 100M words)", "#f2b705"),
      "reddit": ("D", "AskReddit Jan 2018 (139M words)", "#8e3fb5"), "coca": ("^", "COCA (American, 1B words)", "#1a9e8f")}
off = {"bnc": -0.27, "ententen21": -0.09, "reddit": 0.09, "coca": 0.27}

L = pd.read_pickle(config.RESULTS / f"lines_{sc}.pkl"); L = L[L.agg_score.notna()]
B = {c: g.agg_score.values for c, g in L[L.item == BASE].groupby("corpus")}
G = {}
for lab_, (it, fr) in FRAMES.items():
    g = L[(L.item == it) & (L.frame == fr)]
    for c, gg in g.groupby("corpus"): G[(c, lab_)] = gg.agg_score.values

rows = []
for (c, lab_), v in G.items():
    if len(v) < MIN_N: continue
    d, lo, hi, p, dd = welch(v, B[c])
    rows.append(dict(frame=lab_, corpus=c, n=len(v), mean=v.mean(), median=np.median(v), baseline_n=len(B[c]), baseline=B[c].mean(), diff=d, ci_low=lo, ci_high=hi, p=p, cohens_d=dd))
T = pd.DataFrame(rows); T["p_holm"] = holm(T.p.values)
W = T.pivot(index="frame", columns="corpus", values="mean"); M = T.pivot(index="frame", columns="corpus", values="median")
S = T.pivot(index="frame", columns="corpus", values="p_holm"); N = T.pivot(index="frame", columns="corpus", values="n")
items = list(FRAMES)
for lab_ in items: assert all(c in W.columns and not np.isnan(W.loc[lab_, c]) for c in MK), f"{lab_}: a corpus has < {MIN_N} lines"
mean4 = W.mean(axis=1).loc[items].sort_values(); items = list(mean4.index); x = np.arange(len(items))

fig, ax = plt.subplots(figsize=(9.5, 7.4))
HALF = 0.40; wm = {}
for xi, it in zip(x, items):
    cs = list(MK)
    data = np.concatenate([G[(c, it)] for c in cs]); w = np.concatenate([np.full(len(G[(c, it)]), 1 / len(cs) / len(G[(c, it)])) for c in cs])
    wm[it] = np.average(data, weights=w); assert abs(wm[it] - mean4[it]) < 1e-4, (it, wm[it], mean4[it])
    kde = gaussian_kde(data, weights=w, bw_method=0.25)
    g = np.linspace(data.min(), data.max(), 300); d = kde(g); d = d / d.max() * HALF
    ax.fill_betweenx(g, xi - d, xi + d, facecolor=col(wm[it]), edgecolor=INK, linewidth=0.5, zorder=2)
    ax.hlines(wm[it], xi - HALF * 0.9, xi + HALF * 0.9, color=INK, lw=1.6, zorder=3)
for c, (mk, name, fc) in MK.items():
    y = W[c].reindex(items).values; sig = (S[c].reindex(items) < 0.05).values
    ax.scatter((x + off[c])[sig], y[sig], marker=mk, s=46, facecolor=fc, edgecolor=INK, linewidths=0.6, zorder=5, label=name)
    ax.scatter((x + off[c])[~sig], y[~sig], marker=mk, s=46, facecolor="white", edgecolor=fc, linewidths=1.3, zorder=5)
    ax.hlines(M[c].reindex(items).values, x + off[c] - 0.07, x + off[c] + 0.07, color=fc, lw=2.4, zorder=4)
    bl = B[c].mean()
    ax.axhline(bl, color=fc, lw=1.0, ls="--", zorder=1, label=f"{name.split(' (')[0]} baseline, all other verbs: {bl:.2f}")
ax.axhline(5, color=MUTED, lw=0.7, ls=":", zorder=0)
ax.set_xticks(x); ax.set_xticklabels([i for i in items], fontsize=12, fontstyle="italic")
for xi, it in zip(x, items):
    ax.text(xi, -0.075, "n = " + " / ".join(f"{int(N.loc[it, c]):,}" for c in ["ententen21", "bnc", "coca", "reddit"]), ha="center", va="top", fontsize=6.6, color=MUTED, transform=ax.get_xaxis_transform())
ax.set_ylim(1, 10.7); ax.set_yticks(range(1, 10)); ax.set_xlim(-0.6, len(items) - 0.4)
ax.set_ylabel(f"{SCALE} of the effect (per line)", fontsize=8.5, color=MUTED)
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.tick_params(colors=MUTED, labelsize=8)
h, l = ax.get_legend_handles_labels()
h.append(Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor=MUTED)); l.append("hollow marker = not significantly different from that corpus's baseline (Holm)")
h.append(Line2D([], [], color=INK, lw=1.6)); l.append("mean of the four corpora (each 25%)")
h.append(Line2D([], [], color=MUTED, lw=2.4)); l.append("short bar in the corpus colour = that corpus's median")
ax.legend(h, l, loc="upper left", fontsize=7.4, frameon=True, framealpha=0.95, edgecolor="none", ncol=2)
ax.set_title(f"Effect sentiment of the frames cause N to V, force N to V, make N V and let N V, per line ({LABEL[sc]};\nviolin = four corpora weighted equally, markers = each corpus)", fontsize=10, loc="left")
foot = (f"Violin = distribution of the line scores of that frame only, each corpus weighted 25% (weighted kernel density, all violins drawn to the same width). "
        f"Colour: grey = between the lowest and highest of the four corpora's baselines ({LO:.2f}–{HI:.2f}), fading to red below and to green above; dotted line = 5. "
        "Filled marker = significantly different from that corpus's baseline (Welch t-test, Holm-corrected over the 16 frame x corpus tests, p < .05). "
        "Frames: cause / force N to V = 'V + N + to V'; make / let N V = 'V + N + bare V' (noun and verb of the effect averaged; positive and negative forms combined). "
        "Baselines = every other lexical verb, all frames, same rules as in the 101-expression figure. n = scored lines (enTenTen21 / BNC / COCA / AskReddit).")
fig.text(0.01, 0.01, textwrap.fill(foot, 165), fontsize=7, color=MUTED, va="bottom")
fig.subplots_adjust(left=0.1, right=0.99, top=0.92, bottom=0.2)
fig.savefig(out, dpi=200); fig.savefig(out.replace(".png", ".pdf"))
T.assign(weighted_mean_4corpora=T.frame.map(wm)).sort_values(["frame", "corpus"]).to_csv(out.replace(".png", ".tsv"), sep="\t", index=False, float_format="%.5g")
print("saved", out); print(T.sort_values(["frame", "corpus"]).to_string(float_format=lambda v: f"{v:.4g}")); print(wm)
