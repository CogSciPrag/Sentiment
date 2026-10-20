"""Robustness check with a second lexicon, Warriner, Kuperman & Brysbaert (2013) valence norms (V.Mean.Sum, 1-9):
Hedonometer vs Warriner means per corpus, and the main figure re-drawn with Warriner scores.
Input: data/results/extra/ (step 10). Output: data/figures/. Usage: python figures/fig_warriner.py
"""
import numpy as np, pandas as pd, matplotlib
from causesent import config
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from causesent.targets import PREVENTION, EFFECT_LEFT
EXTRA = str(config.RESULTS / "extra") + "/"; FIG = str(config.FIGURES) + "/"; config.ensure_dirs()
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
CMAP = LinearSegmentedColormap.from_list("rg", ["#b2182b", "#e8816b", "#dcdad4", "#7fbf7b", "#1b7837"])
NORM = TwoSlopeNorm(vmin=3.5, vcenter=5.0, vmax=6.5); col = lambda v: CMAP(NORM(min(max(v, 3.5), 6.5)))
MK = {"ententen21": ("o", "enTenTen21 (web, 52B words)", "#2b6cd4"), "bnc": ("s", "BNC (British, 100M words)", "#f2b705"), "reddit": ("D", "AskReddit Jan 2018 (139M words)", "#8e3fb5")}
KEY = ["cause", "make", "ensure", "guarantee", "secure", "prevent", "trigger", "foster", "lead to", "result in"]
lab = lambda it: it + (" *" if it == "make" else "") + (" †" if it in PREVENTION else "") + (" ←" if it in EFFECT_LEFT else "")
S = pd.read_csv(EXTRA + "word_stats.tsv", sep="\t")
tidy = lambda ax: ([ax.spines[s].set_visible(False) for s in ["top", "right"]], ax.tick_params(colors=MUTED, labelsize=8))

fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
for ax, (c, (mk, name, fc)) in zip(axes, MK.items()):
    h = S[(S.corpus == c) & (S.scoring == "raw")].set_index("item")["mean"]; w = S[(S.corpus == c) & (S.scoring == "war")].set_index("item")["mean"]
    j = pd.concat([h, w], axis=1, keys=["h", "w"]).dropna()
    ax.plot([3.8, 6.6], [3.8, 6.6], color=MUTED, lw=0.7, ls="--")
    ax.scatter(j.h, j.w, marker=mk, s=26, facecolor=fc, edgecolor=INK, linewidths=0.5, zorder=3)
    for it in KEY:
        if it in j.index: ax.annotate(lab(it), (j.h[it], j.w[it]), xytext=(4, -3), textcoords="offset points", fontsize=7.5, fontstyle="italic")
    ax.set_xlim(3.8, 6.6); ax.set_ylim(3.8, 6.6); ax.set_aspect("equal"); ax.grid(color=GRID, lw=0.6); tidy(ax)
    ax.set_xlabel("Hedonometer (labMT), raw", fontsize=8.5, color=MUTED); ax.set_ylabel("Warriner et al. (2013) valence", fontsize=8.5, color=MUTED)
    ax.set_title(f"{name.split(' (')[0]}: {len(j)} words, Spearman ρ = {spearmanr(j.h, j.w)[0]:.2f}", fontsize=10, loc="left")
fig.suptitle("Robustness check: word means under the Hedonometer and under Warriner et al.'s valence norms", fontsize=13, x=0.01, ha="left")
fig.text(0.01, 0.012, "Each point = one word's mean effect score (combined N+V, same lines, same frame rules). Dashed line = equal scores. Both scales run 1–9 with 5 neutral.",
         fontsize=7.4, color=MUTED)
fig.tight_layout(rect=(0, 0.06, 1, 0.94)); fig.savefig(FIG + "fig7_hedonometer_vs_warriner.png", dpi=200); fig.savefig(FIG + "fig7_hedonometer_vs_warriner.pdf"); plt.close(fig)

# fig8: figure C with Warriner
W = S[S.scoring == "war"].pivot(index="item", columns="corpus", values="mean")
mean3 = W.mean(axis=1).sort_values(); items = list(mean3.index); x = np.arange(len(items))
base = S[(S.scoring == "war") & (S.corpus == "reddit")].baseline.iloc[0]
fig, ax = plt.subplots(figsize=(17, 7.8))
ax.bar(x, mean3.values - 1, bottom=1, color=[col(v) for v in mean3.values], edgecolor=INK, linewidth=0.4, width=0.78, zorder=2)
off = {"ententen21": 0, "bnc": -0.2, "reddit": 0.2}
for c, (mk, name, fc) in MK.items():
    ax.scatter(x + off[c], W[c].reindex(items).values, marker=mk, s=20, facecolor=fc, edgecolor=INK, linewidths=0.6, zorder=4, label=name)
ax.axhline(5, color=MUTED, lw=0.7, ls=":")
ax.axhline(base, color="#8e3fb5", lw=1.1, ls="--", zorder=3, label=f"AskReddit baseline, all other verbs (3M words): {base:.2f}")
ax.set_xticks(x); ax.set_xticklabels([lab(i) for i in items], rotation=90, fontsize=8.3, fontstyle="italic")
for t, it in zip(ax.get_xticklabels(), items):
    if it in ("cause", "ensure", "guarantee", "secure", "prevent", "trigger", "foster"): t.set_fontweight("bold")
ax.set_ylim(1, 7.6); ax.set_yticks(range(1, 8)); ax.set_xlim(-0.8, len(items) - 0.2); ax.grid(axis="y", color=GRID, lw=0.6); tidy(ax)
ax.set_ylabel("Mean valence of the effect (Warriner et al. 2013)\n(aggregate over all frames)", fontsize=8.5, color=MUTED)
rs = [spearmanr(W[a], W[b], nan_policy="omit")[0] for a, b in [("ententen21", "bnc"), ("ententen21", "reddit"), ("bnc", "reddit")]]
h, l = ax.get_legend_handles_labels(); h.append(Line2D([], [], color="none"))
l.append(f"Rank agreement (Spearman): enTenTen21–BNC {rs[0]:.2f}, enTenTen21–Reddit {rs[1]:.2f}, BNC–Reddit {rs[2]:.2f}")
ax.legend(h, l, loc="upper left", fontsize=7.8, frameon=True, framealpha=0.95, edgecolor="none")
ax.set_title("Effect sentiment of 78 causal words scored with Warriner et al.'s (2013) valence norms instead of the Hedonometer", fontsize=11, loc="left")
fig.text(0.01, 0.005, "Same lines and frame rules as figure C (make: causative frames only; passives and negated effects excluded); only means with ≥ 30 scored lines. "
         "Bars = mean of the three corpora; markers = each corpus. Bars start at 1.\nWarriner's norms rate 13,915 lemmas (1 = unhappy, 9 = happy) but leave out function words such as pronouns, so fewer lines are scored "
         "(Reddit 81% vs 92%). † prevention; ← effect on the left; * make: causative frames only.", fontsize=7.4, color=MUTED, va="bottom")
fig.subplots_adjust(left=0.06, right=0.99, top=0.94, bottom=0.27)
fig.savefig(FIG + "fig8_warriner_effect_sentiment.png", dpi=200); fig.savefig(FIG + "fig8_warriner_effect_sentiment.pdf"); print("ok")
