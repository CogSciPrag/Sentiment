"""Figures from the extra analyses (step 10): ranking of each expression among all other verbs (AskReddit), share of
clearly negative / neutral / positive effects per expression, score distributions for key expressions, and word-group means.
Input: data/results/extra/ (step 10). Output: data/figures/. Usage: python figures/fig_distributions_and_groups.py
"""
import numpy as np, pandas as pd, matplotlib
from causesent import config
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.stats import gaussian_kde
from causesent.targets import PREVENTION, EFFECT_LEFT
EXTRA = str(config.RESULTS / "extra") + "/"; FIG = str(config.FIGURES) + "/"; config.ensure_dirs()

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
CMAP = LinearSegmentedColormap.from_list("rg", ["#b2182b", "#e8816b", "#dcdad4", "#7fbf7b", "#1b7837"])
NORM = TwoSlopeNorm(vmin=3.5, vcenter=5.0, vmax=6.5)
col = lambda v: CMAP(NORM(min(max(v, 3.5), 6.5)))
KEY = ["cause", "make", "ensure", "guarantee", "secure", "prevent", "trigger", "foster", "lead to", "result in"]
MK = {"ententen21": ("o", "enTenTen21", "#2b6cd4"), "bnc": ("s", "BNC", "#f2b705"), "reddit": ("D", "AskReddit", "#8e3fb5")}
lab = lambda it: it + (" *" if it == "make" else "") + (" †" if it in PREVENTION else "") + (" ←" if it in EFFECT_LEFT else "")
SC = {"raw": "raw scores", "band": "neutral band (4–6) removed"}
S = pd.read_csv(EXTRA + "word_stats.tsv", sep="\t")
def tidy(ax):
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
def save(fig, name):
    fig.savefig(f"{FIG}{name}.png", dpi=200); fig.savefig(f"{FIG}{name}.pdf"); plt.close(fig)

# ---- Fig 1: where the 78 words fall among all other verbs (AskReddit)
fig, axes = plt.subplots(2, 1, figsize=(15, 9.5))
for ax, sc in zip(axes, ["raw", "band"]):
    vm = pd.read_csv(f"{EXTRA}other_verb_means_{sc}.tsv", sep="\t")
    bins = np.arange(2.0, 7.61, 0.1)
    ax.hist(vm["mean"], bins=bins, color="#cfcdc6", edgecolor="white", zorder=2, label=f"other verbs (n = {len(vm)} verbs with ≥ 30 scored lines)")
    r = S[(S.scoring == sc) & (S.corpus == "reddit")].sort_values("mean")
    ymax = ax.get_ylim()[1]
    kk = 0
    for i, (_, w) in enumerate(r.iterrows()):
        k = w["item"] in KEY
        h = ymax * (1.0 + 0.16 * (kk % 4)) if k else ymax * 0.12
        ax.plot([w["mean"]] * 2, [0, h], color=col(w["mean"]), lw=1.6 if k else 0.8, zorder=3)
        if k:
            ax.text(w["mean"], h, f" {lab(w['item'])} ({w['pct_verbs_below']:.0f}%)", fontsize=8,
                    fontstyle="italic", ha="left", va="center", color=INK); kk += 1
    base = r["baseline"].iloc[0]
    ax.axvline(base, color="#8e3fb5", ls="--", lw=1.1, zorder=4, label=f"pooled baseline = {base:.2f}")
    ax.set_xlim(2.0, 7.6); ax.set_ylim(0, ymax * 1.75)
    ax.set_xlabel(f"Mean Hedonometer score of the effect (combined N+V), {SC[sc]}", fontsize=8.5, color=MUTED)
    ax.set_ylabel("number of verbs", fontsize=8.5, color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0); tidy(ax)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_title(f"{'A' if sc == 'raw' else 'B'}. AskReddit, {SC[sc]}: short ticks = all 77 of our words; tall ticks = key words "
                 "(in brackets: % of other verbs with a lower mean)", fontsize=10, loc="left")
fig.suptitle("Where the causal words fall among all other English verbs (AskReddit Jan 2018, 3M-word baseline sample)", fontsize=13, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.97)); save(fig, "fig1_rank_among_verbs")

# ---- Fig 2: difference from the baseline with 95% CI (AskReddit), raw and band
r = S[(S.corpus == "reddit")]
order = r[r.scoring == "raw"].sort_values("diff").item.tolist()
fig, axes = plt.subplots(1, 2, figsize=(13, 17), sharey=True)
for ax, sc in zip(axes, ["raw", "band"]):
    q = r[r.scoring == sc].set_index("item").reindex(order)
    y = np.arange(len(order))
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.errorbar(q["diff"], y, xerr=[q["diff"] - q.ci_low, q.ci_high - q["diff"]], fmt="none", ecolor=MUTED, elinewidth=0.8)
    ax.scatter(q["diff"], y, s=26, c=[col(5 + d) for d in q["diff"].fillna(0)], edgecolors=INK, linewidths=0.5, zorder=3)
    for yi, (it, w) in zip(y, q.iterrows()):
        if pd.notna(w.p_holm) and w.p_holm >= 0.05: ax.text(w.ci_high + 0.04, yi, "n.s.", fontsize=6.5, color=MUTED, va="center")
    ax.set_yticks(y); ax.set_yticklabels([lab(i) for i in order], fontsize=7.8, fontstyle="italic"); ax.set_ylim(len(order) - 0.3, -0.7)
    ax.set_xlabel(f"Difference from the baseline (Hedonometer points), {SC[sc]}", fontsize=8.5, color=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.6); tidy(ax)
    ax.set_title(f"{'A' if sc == 'raw' else 'B'}. {SC[sc].capitalize()} (baseline = {q.baseline.dropna().iloc[0]:.2f})", fontsize=10, loc="left")
for t in axes[0].get_yticklabels():
    if t.get_text().split(" ")[0] in ("cause", "ensure", "guarantee", "secure", "prevent", "trigger", "foster"): t.set_fontweight("bold")
fig.suptitle("Each word's effect sentiment compared with all other verbs (AskReddit)", fontsize=13, x=0.01, ha="left")
fig.text(0.01, 0.004, "Points = word mean minus the pooled baseline of all other verbs (same text, same frame rules). Bars = 95% CI (Welch). "
         "n.s. = not significant after Holm correction for 77 comparisons.\nSorted by the raw difference. † prevention words; ← effect on the left; * make: causative frames only.",
         fontsize=7.4, color=MUTED, va="bottom")
fig.tight_layout(rect=(0, 0.025, 1, 0.975)); save(fig, "fig2_difference_from_baseline")

# ---- Fig 3: distribution of effect scores: share clearly negative / neutral / clearly positive (raw), all corpora
A = pd.read_pickle(EXTRA + "all_lines.pkl")
fig = plt.figure(figsize=(17, 15))
gs = fig.add_gridspec(2, 3, height_ratios=[2.3, 1], hspace=0.35, wspace=0.25)
order3 = S[(S.scoring == "raw") & (S.corpus == "ententen21")].sort_values("pct_below4", ascending=False).item.tolist()
for j, corp in enumerate(["ententen21", "bnc", "reddit"]):
    ax = fig.add_subplot(gs[0, j])
    q = S[(S.scoring == "raw") & (S.corpus == corp)].set_index("item").reindex(order3)
    y = np.arange(len(order3))
    ax.barh(y, q.pct_below4, color="#c84b4b", height=0.8, label="clearly negative (< 4)")
    ax.barh(y, q.pct_4to6, left=q.pct_below4, color="#e3e1db", height=0.8, label="4–6")
    ax.barh(y, q.pct_above6, left=q.pct_below4 + q.pct_4to6, color="#4f9a55", height=0.8, label="clearly positive (> 6)")
    if corp == "reddit":
        b = A[(A.item == "BASELINE (other verbs)") & (A.corpus == "reddit")].agg_raw.dropna()
        ax.axvline((b < 4).mean() * 100, color="#8e3fb5", ls="--", lw=1); ax.axvline(100 - (b > 6).mean() * 100, color="#8e3fb5", ls="--", lw=1)
        ax.text((b < 4).mean() * 100 + 1, -1.2, f"baseline {(b < 4).mean()*100:.0f}% < 4", fontsize=7, color="#8e3fb5")
        ax.text(100 - (b > 6).mean() * 100 - 1, -1.2, f"baseline {(b > 6).mean()*100:.0f}% > 6", fontsize=7, color="#8e3fb5", ha="right")
    ax.set_yticks(y); ax.set_yticklabels([lab(i) for i in order3] if j == 0 else [], fontsize=7.2, fontstyle="italic")
    ax.set_ylim(len(order3) - 0.4, -1.8); ax.set_xlim(0, 100); tidy(ax)
    ax.set_xlabel("% of scored lines", fontsize=8.5, color=MUTED)
    ax.set_title(f"{'ABC'[j]}. {MK[corp][1]}", fontsize=10, loc="left")
    if j == 2: ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.07), ncol=3, fontsize=7.5, frameon=False)
# densities
for j, corp in enumerate(["ententen21", "bnc", "reddit"]):
    ax = fig.add_subplot(gs[1, j]); xs = np.linspace(1.5, 8.5, 300)
    for it, c in [("cause", "#b2182b"), ("prevent", "#e8816b"), ("make", "#8c8c8c"), ("ensure", "#1b7837"), ("guarantee", "#2b6cd4")]:
        x = A[(A.corpus == corp) & (A.item == it)].agg_raw.dropna()
        ax.plot(xs, gaussian_kde(x, bw_method=0.12)(xs), color=c, lw=1.4, label=lab(it))
    if corp == "reddit":
        b = (lambda b: b.sample(min(50000, len(b)), random_state=1))(A[(A.item == "BASELINE (other verbs)") & (A.corpus == "reddit")].agg_raw.dropna())
        ax.plot(xs, gaussian_kde(b, bw_method=0.12)(xs), color="#8e3fb5", lw=1.4, ls="--", label="baseline (other verbs)")
    ax.axvline(5, color=MUTED, lw=0.6, ls=":"); tidy(ax); ax.set_yticks([])
    ax.set_xlabel("Hedonometer score of the effect (raw, combined N+V)", fontsize=8.5, color=MUTED)
    ax.set_title(f"{'DEF'[j]}. Distribution of line scores, {MK[corp][1]}", fontsize=10, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
fig.suptitle("Distribution of effect sentiment: share of clearly negative and clearly positive effects (raw scores)", fontsize=13, x=0.01, ha="left", y=0.995)
fig.text(0.01, 0.003, "A–C sorted by the share of clearly negative effects in enTenTen21. Only words with ≥ 30 scored lines in that corpus are shown. "
         "Purple dashed lines in C = the AskReddit baseline of all other verbs.", fontsize=7.4, color=MUTED, va="bottom")
fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.05); save(fig, "fig3_distributions")

# ---- Fig 4: provisional word groups
G = pd.read_csv(EXTRA + "group_means.tsv", sep="\t")
fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
gorder = sorted(G.group.unique(), key=lambda g: int(g.split()[0]))
for ax, sc in zip(axes, ["raw", "band"]):
    y = np.arange(len(gorder))
    for corp, (mk, name, fc) in MK.items():
        q = G[(G.scoring == sc) & (G.corpus == corp)].set_index("group").reindex(gorder)
        off = {"ententen21": -0.2, "bnc": 0, "reddit": 0.2}[corp]
        ax.errorbar(q.mean_of_word_means, y + off, xerr=[q.mean_of_word_means - q.ci_low, q.ci_high - q.mean_of_word_means], fmt="none", ecolor=MUTED, elinewidth=0.7)
        ax.scatter(q.mean_of_word_means, y + off, marker=mk, s=30, facecolor=fc, edgecolor=INK, linewidths=0.5, zorder=3, label=name)
    base = S[(S.scoring == sc) & (S.corpus == "reddit")].baseline.iloc[0]
    ax.axvline(base, color="#8e3fb5", ls="--", lw=1.1, label=f"AskReddit baseline = {base:.2f}")
    ax.axvline(5, color=MUTED, lw=0.6, ls=":")
    ax.set_yticks(y); ax.set_yticklabels(gorder, fontsize=8.5); ax.set_ylim(len(gorder) - 0.4, -0.6)
    ax.grid(axis="x", color=GRID, lw=0.6); tidy(ax)
    ax.set_xlabel(f"Mean of the word means in the group, {SC[sc]}", fontsize=8.5, color=MUTED)
    ax.set_title(f"{'A' if sc == 'raw' else 'B'}. {SC[sc].capitalize()}", fontsize=10, loc="left")
axes[0].legend(fontsize=7.8, loc="lower left", frameon=True, framealpha=0.95, edgecolor="none")
fig.suptitle("Effect sentiment by (provisional) word group", fontsize=13, x=0.01, ha="left")
fig.text(0.01, 0.005, "Each word counts once (unweighted mean of word means); bars = 95% bootstrap CI over the words in the group. "
         "Groups are provisional; see group_means.tsv for the words in each group.", fontsize=7.4, color=MUTED, va="bottom")
fig.tight_layout(rect=(0, 0.03, 1, 0.96)); save(fig, "fig4_word_groups")
print("figs done")
