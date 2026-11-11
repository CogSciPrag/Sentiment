"""Portrait (A4) version of the main figure: mean effect sentiment of each studied expression in each corpus.

Same data, colours and markers as fig_effect_sentiment.py, but drawn as horizontal bars, one row per expression,
lowest mean at the top; 5.6 x 11.69 in (A4 height, narrow), x-axis starts at 3 (5 Nov 2026).
Bar = mean of the corpora's means; markers = each corpus (filled = significantly different from that corpus's
baseline of all other verbs, Welch t-test, Holm-corrected; hollow = not significant); dashed lines = baselines.
Input: data/results/pvalues_all.tsv (step 9). Output: data/figures/fig_effect_sentiment_<scoring>_portrait.png and .pdf
Usage: python figures/fig_effect_sentiment_portrait.py [scoring]     (raw, band, adj or war; default raw)
"""
import sys, textwrap
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from causesent import config
import causesent.style  # Helvetica
from causesent.targets import PREVENTION, EFFECT_LEFT, SYMMETRIC, DISPLAY
from causesent.palette import centre, bounds, band_colour_fn

sc = sys.argv[1] if len(sys.argv) > 1 else "raw"
config.ensure_dirs(); out = str(config.FIGURES / f"fig_effect_sentiment_{sc}_portrait.png")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
LO, HI = bounds(sc); col = band_colour_fn(LO, HI)   # grey band = lowest to highest of the four corpora's baselines
MIN_N = 30
LABEL = {"raw": "raw scores", "band": "neutral band (4–6) removed", "adj": "adjectives and noun compounds averaged into the noun",
         "war": "Warriner et al. (2013) valence norms"}
SCALE = "Warriner valence" if sc == "war" else "Hedonometer score"
MK = {"ententen21": ("o", "enTenTen21 (web, 52B words)", "#2b6cd4"), "bnc": ("s", "BNC (British, 100M words)", "#f2b705"),
      "reddit": ("D", "AskReddit Jan 2018 (139M words)", "#8e3fb5"), "coca": ("^", "COCA (American, 1B words)", "#1a9e8f")}
lab = lambda it: DISPLAY.get(it, it) + (" ←" if it in EFFECT_LEFT else "")   # only the left arrow (effect on the left); no † * ≈ (5 Nov 2026)

P = pd.read_csv(config.RESULTS / "pvalues_all.tsv", sep="\t"); P = P[(P.scoring == sc) & (P.measure == "NV")]
if "polarity" in P: P = P[P.polarity == "combined"]
if "baseline_type" in P: P = P[P.baseline_type == "verbs"]
Q = P[P.n >= MIN_N]
W = Q.pivot(index="item", columns="corpus", values="mean"); S = Q.pivot(index="item", columns="corpus", values="p_holm")
mean3 = W.mean(axis=1).sort_values()          # lowest first = top row
items = list(mean3.index); y = np.arange(len(items))

W_IN, H_IN = 5.6, 11.69
fig = plt.figure(figsize=(W_IN, H_IN))
X0 = 3.0                                                      # the axis (and the bars) start at 3
ax = fig.add_axes([1.12 / W_IN, 1.5 / H_IN, (W_IN - 1.12 - 0.12) / W_IN, (H_IN - 1.5 - 1.32) / H_IN])
ax.barh(y, mean3.values - X0, left=X0, color=[col(m) for m in mean3.values], edgecolor=INK, linewidth=0.3, height=0.8, zorder=2)
off = {"bnc": -0.27, "ententen21": -0.09, "reddit": 0.09, "coca": 0.27}
for c, (mk, name, fc) in MK.items():
    if c not in W: continue
    v = W[c].reindex(items).values; sig = (S[c].reindex(items) < 0.05).values
    ax.scatter(v[sig], (y + off[c])[sig], marker=mk, s=9, facecolor=fc, edgecolor=INK, linewidths=0.4, zorder=4)
    ax.scatter(v[~sig], (y + off[c])[~sig], marker=mk, s=9, facecolor="white", edgecolor=fc, linewidths=0.7, zorder=4)
    bl = P[P.corpus == c].baseline.dropna()
    if len(bl): ax.axvline(bl.iloc[0], color=fc, lw=0.8, ls="--", zorder=3)
ax.axvline(5, color=MUTED, lw=0.6, ls=":", zorder=1)
ax.set_yticks(y); ax.set_yticklabels([lab(i) for i in items], fontsize=6.2, fontstyle="italic", color=INK)
ax.tick_params(axis="y", length=1.5, pad=1.5, colors=MUTED, labelcolor=INK); ax.tick_params(axis="x", labelsize=7, colors=MUTED)
ax.set_ylim(len(items) - 0.4, -0.6); ax.set_xlim(X0, 6.5); ax.set_xticks(range(3, 7))
ax.set_xlabel(f"Mean {SCALE} of the effect (aggregate over all frames)", fontsize=7, color=MUTED)
ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
for s in ["top", "right"]: ax.spines[s].set_visible(False)

def _rho(a, b):
    if a not in W or b not in W: return float("nan")
    j = W[[a, b]].dropna()
    return spearmanr(j[a], j[b])[0] if len(j) >= 3 else float("nan")
PAIRS = [("ententen21", "bnc"), ("ententen21", "reddit"), ("bnc", "reddit"), ("coca", "ententen21"), ("coca", "bnc"), ("coca", "reddit")]
rs = [_rho(a, b) for a, b in PAIRS]

from matplotlib.legend_handler import HandlerTuple
h, l = [], []
for c, (mk, name, fc) in MK.items():
    bl = P[P.corpus == c].baseline.dropna()
    h.append((Line2D([], [], color=fc, lw=1.0, ls="--"), Line2D([], [], marker=mk, color="none", markerfacecolor=fc, markeredgecolor=INK, markeredgewidth=0.5, markersize=4.5)))
    l.append(f"{name}; baseline of all other verbs {bl.iloc[0]:.2f}" if len(bl) else name)
h.append(Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor=MUTED, markersize=4.5)); l.append("hollow marker = not significantly different from that corpus's baseline (Holm)")
fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.02, 1 - 0.55 / H_IN), fontsize=6, frameon=False, ncol=1, labelspacing=0.45, handletextpad=0.5, handlelength=2.6,
           handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)})

_nc = sum(i in SYMMETRIC for i in mean3.dropna().index); _n = len(mean3.dropna()) - _nc
fig.text(0.02, 0.992, f"Effect sentiment of {_n + _nc} expressions ({LABEL[sc]})", fontsize=10, va="top", color=INK)
fig.text(0.02, 0.976, "Bar = mean of the corpora; markers = each corpus.", fontsize=7.5, va="top", color=MUTED)

note = (f"Colour: grey = between the lowest and highest of the four corpora's baselines ({LO:.2f}–{HI:.2f}), fading to red below and to green above; bars start at 3; dotted line = 5. "
        "Only means with ≥ 30 scored lines. Filled marker = significantly different from that corpus's baseline (Welch t-test, Holm-corrected, p < .05). "
        "Baselines = every other lexical verb, same rules: a random 10,000-line Sketch Engine sample (enTenTen21, BNC), a random 3M-word sample (AskReddit), every lexical verb in 5,500 random COCA lines around 11 function words (COCA). "
        "COCA expressions: 500 random lines per query (fewer for rare queries; items searched with several queries combined by frequency; 'as' not in COCA). "
        "← effect on the left.  " + "Prevention words (" + ", ".join(i for i in items if i in PREVENTION and i != "fail to") + ") and 'fail to': the complement is what does NOT happen.  "
        + "Correlation items (" + ", ".join(i for i in items if i in SYMMETRIC) + "; non-causal comparison): both sides scored, X as N and Y as V.  "
        "make and get: causative frames only (make N ADJ / N V / N V-ed; get N ADJ / N V-ed / N to V).  "
        "determine: causal sense only (explicit 'determine that' + clause is excluded as epistemic).  "
        f"Rank agreement (Spearman): enT–BNC {rs[0]:.2f}, enT–Reddit {rs[1]:.2f}, BNC–Reddit {rs[2]:.2f}, COCA–enT {rs[3]:.2f}, COCA–BNC {rs[4]:.2f}, COCA–Reddit {rs[5]:.2f}.")
fig.text(0.02, 0.012, textwrap.fill(note, 148), fontsize=5.4, color=MUTED, va="bottom", linespacing=1.35)
fig.savefig(out, dpi=300); fig.savefig(out.replace(".png", ".pdf")); print("saved", out)
