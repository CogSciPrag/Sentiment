"""Main figure: mean effect sentiment of each studied expression in each corpus, against each corpus's baseline.

Bar = mean of the corpora's means; markers = each corpus (filled = significantly different from that corpus's
baseline of all other verbs, Welch t-test, Holm-corrected; hollow = not significant); dashed lines = baselines.
Input: data/results/pvalues_all.tsv (step 9). Output: data/figures/fig_effect_sentiment_<scoring>.png and .pdf
Usage: python figures/fig_effect_sentiment.py [scoring]     (raw, band, adj or war; default raw)
"""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from causesent import config
from causesent.targets import PREVENTION, EFFECT_LEFT, SYMMETRIC

sc = sys.argv[1] if len(sys.argv) > 1 else "raw"
config.ensure_dirs(); out = str(config.FIGURES / f"fig_effect_sentiment_{sc}.png")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
CMAP = LinearSegmentedColormap.from_list("rg", ["#b2182b", "#e8816b", "#dcdad4", "#7fbf7b", "#1b7837"])
NORM = TwoSlopeNorm(vmin=3.5, vcenter=5.0, vmax=6.5); col = lambda v: CMAP(NORM(min(max(v, 3.5), 6.5)))
MIN_N = 30
LABEL = {"raw": "raw scores", "band": "neutral band (4–6) removed", "adj": "adjectives and noun compounds averaged into the noun",
         "war": "Warriner et al. (2013) valence norms"}
SCALE = "Warriner valence" if sc == "war" else "Hedonometer score"
MK = {"ententen21": ("o", "enTenTen21 (web, 52B words)", "#2b6cd4"), "bnc": ("s", "BNC (British, 100M words)", "#f2b705"),
      "reddit": ("D", "AskReddit Jan 2018 (139M words)", "#8e3fb5"), "coca": ("^", "COCA (American, 1B words)", "#1a9e8f")}
KEY = {"cause", "ensure", "guarantee", "secure", "prevent", "trigger", "foster", "associated with", "correlated with"}
lab = lambda it: it + (" *" if it in ("make", "get") else "") + (" †" if it in PREVENTION else "") + (" ←" if it in EFFECT_LEFT else "") + (" ≈" if it in SYMMETRIC else "")
def tidy(ax):
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)

P = pd.read_csv(config.RESULTS / "pvalues_all.tsv", sep="\t"); P = P[(P.scoring == sc) & (P.measure == "NV")]

fig, ax = plt.subplots(figsize=(17, 7.5))
Q = P[P.n >= MIN_N]
W = Q.pivot(index="item", columns="corpus", values="mean"); S = Q.pivot(index="item", columns="corpus", values="p_holm")
mean3 = W.mean(axis=1).sort_values()   # mean over the corpora
items = list(mean3.index); x = np.arange(len(items))
ax.bar(x, mean3.values - 1, bottom=1, color=[col(m) for m in mean3.values], edgecolor=INK, linewidth=0.4, width=0.78, zorder=2)
off = {"bnc": -0.27, "ententen21": -0.09, "reddit": 0.09, "coca": 0.27}
for c, (mk, name, fc) in MK.items():
    if c not in W: continue
    y = W[c].reindex(items).values; sig = (S[c].reindex(items) < 0.05).values
    ax.scatter((x + off[c])[sig], y[sig], marker=mk, s=20, facecolor=fc, edgecolor=INK, linewidths=0.6, zorder=4, label=name)
    ax.scatter((x + off[c])[~sig], y[~sig], marker=mk, s=20, facecolor="white", edgecolor=fc, linewidths=1.0, zorder=4)
    bl = P[P.corpus == c].baseline.dropna()
    if len(bl): ax.axhline(bl.iloc[0], color=fc, lw=1.0, ls="--", zorder=3, label=f"{name.split(' (')[0]} baseline, all other verbs: {bl.iloc[0]:.2f}")
ax.axhline(5, color=MUTED, lw=0.7, ls=":", zorder=1)
ax.set_xticks(x); ax.set_xticklabels([lab(i) for i in items], rotation=90, fontsize=8.3, fontstyle="italic")
for t, it in zip(ax.get_xticklabels(), items):
    if it in KEY: t.set_fontweight("bold")
ax.set_ylim(1, 7.8); ax.set_yticks(range(1, 8)); ax.set_xlim(-0.8, len(items) - 0.2)
ax.set_ylabel(f"Mean {SCALE} of the effect\n(aggregate over all frames)", fontsize=8.5, color=MUTED)
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0); tidy(ax)
def _rho(a, bb):
    if a not in W or bb not in W: return float("nan")
    j = W[[a, bb]].dropna()
    return spearmanr(j[a], j[bb])[0] if len(j) >= 3 else float("nan")
PAIRS = [("ententen21", "bnc"), ("ententen21", "reddit"), ("bnc", "reddit"), ("coca", "ententen21"), ("coca", "bnc"), ("coca", "reddit")]
rs = [_rho(a, bb) for a, bb in PAIRS]
h, l = ax.get_legend_handles_labels()
h.append(Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor=MUTED)); l.append("hollow marker = not significantly different from that corpus's baseline (Holm)")
h.append(Line2D([], [], color="none")); l.append(f"Rank agreement (Spearman): enT–BNC {rs[0]:.2f}, enT–Reddit {rs[1]:.2f}, BNC–Reddit {rs[2]:.2f}, COCA–enT {rs[3]:.2f}, COCA–BNC {rs[4]:.2f}, COCA–Reddit {rs[5]:.2f}")
ax.legend(h, l, loc="upper left", fontsize=7.4, frameon=True, framealpha=0.95, edgecolor="none", ncol=2)
ax.set_title(f"Effect sentiment of 78 causal expressions and 2 correlation expressions ({LABEL[sc]}; bar = mean of the corpora, markers = each corpus)", fontsize=10.5, loc="left")

fig.text(0.01, 0.01,
         "Colour: red = negative, grey = neutral (5), green = positive; bars start at 1; dotted line = neutral (5). Only means with ≥ 30 scored lines. "
         "Filled marker = significantly different from that corpus's baseline (Welch t-test, Holm-corrected, p < .05).\n"
         "Baselines = every other lexical verb, same rules: a random 10,000-line Sketch Engine sample (enTenTen21, BNC), a random 3M-word sample (AskReddit), every lexical verb in 5,500 random COCA lines around 11 function words (COCA). COCA expressions: 500 random lines each.\n"
         "† prevention words and 'fail to': the complement is what does NOT happen.   ← effect on the left.   ≈ correlation (non-causal comparison): both sides scored, X as N and Y as V.   * make and get: causative frames only (make N ADJ / N V / N V-ed; get N ADJ / N V-ed / N to V).",
         fontsize=7.2, color=MUTED, va="bottom")
fig.subplots_adjust(left=0.055, right=0.99, top=0.93, bottom=0.3)
fig.savefig(out, dpi=200); fig.savefig(out.replace(".png", ".pdf")); print("saved", out)
