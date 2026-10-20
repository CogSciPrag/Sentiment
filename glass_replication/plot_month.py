"""Figure: Glass's own pipeline on all of AskReddit, January 2018 (139M words), effect-side measures only."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from causesent import config
config.ensure_dirs()

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
CMAP = LinearSegmentedColormap.from_list("rg", ["#b2182b", "#e8816b", "#dcdad4", "#7fbf7b", "#1b7837"])
NORM = TwoSlopeNorm(vmin=3.5, vcenter=5.0, vmax=6.5)
def vcol(v): return CMAP(NORM(min(max(v, 3.5), 6.5)))
SUFF = {"cause", "ensure", "guarantee", "secure", "trigger", "spark", "provoke", "prompt", "induce"}
NEC = {"enable", "allow", "let", "permit", "require", "depend", "necessitate"}
CAUSEE = {"induce", "prompt", "permit", "get", "force", "compel", "make", "help", "let", "allow", "enable", "render"}
NOT_EFFECT_OBJ = {"depend", "arise", "stem", "result", "lead", "contribute"}   # effect is not the direct object
def lab(w): return w + (" (S)" if w in SUFF else " (N)" if w in NEC else "") + ("*" if w in CAUSEE else "")

S = pd.read_csv(config.RESULTS / "glass" / "month_summary.tsv", sep="\t")
base = S[S.verb.str.startswith("BASELINE")].iloc[0]
S = S[~S.verb.str.startswith("BASELINE")]

fig, axes = plt.subplots(2, 1, figsize=(13, 9.6), gridspec_kw=dict(height_ratios=[1, 1]))
for ax, m, title in [(axes[0], "dobj", "A. Direct object of the verb (Glass's main measure; single word, as in her code)"),
                     (axes[1], "ccomp", "B. Clausal complement of the verb (e.g. cause [the price to rise], ensure [that ...])")]:
    d = S[(S[m + "_n"] >= 30) & S[m + "_mean"].notna()]
    if m == "dobj": d = d[~d.verb.isin(NOT_EFFECT_OBJ)]
    d = d.sort_values(m + "_mean").reset_index(drop=True)
    x = np.arange(len(d))
    ax.bar(x, d[m + "_mean"] - 1, bottom=1, width=0.72, color=[vcol(v) for v in d[m + "_mean"]], zorder=2)
    ax.errorbar(x, d[m + "_mean"], yerr=1.96 * d[m + "_se"], fmt="none", ecolor=INK, elinewidth=0.8, capsize=1.8, zorder=3)
    for xi, r in d.iterrows():
        ax.text(xi, r[m + "_mean"] + 1.96 * r[m + "_se"] + 0.05, f"{r[m + '_mean']:.2f}", ha="center", va="bottom", fontsize=6.6, color=MUTED, rotation=90)
    ax.axhline(5, color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.axhline(base[m + "_mean"], color="#2a78d6", lw=1.1, ls=":", zorder=1)
    ax.text(-0.5, base[m + "_mean"] + 0.05, f"Glass's baseline: all other verbs ({base[m + '_mean']:.2f})", ha="left", va="bottom", fontsize=7.5, color="#2a78d6")
    ax.set_xticks(x); ax.set_xticklabels([f"{lab(w)}\nn={n:,}" for w, n in zip(d.verb, d[m + "_n"])], rotation=90, fontsize=7.6, fontstyle="italic")
    ax.set_ylim(3, 7.3); ax.set_xlim(-0.7, len(d) - 0.3)
    ax.set_ylabel("Mean Hedonometer score\n(1 = sad … 9 = happy)", fontsize=8.5, color=MUTED)
    ax.set_title(title, fontsize=10, loc="left", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", labelsize=8, colors=MUTED)
fig.suptitle("Glass's own pipeline on all of r/AskReddit, January 2018 (4.5M comments, 139M words)", fontsize=12.5, x=0.01, ha="left", color=INK)
fig.text(0.01, 0.006,
         "Her code unchanged (spaCy en_core_web_sm 3.6.0, one-verb clauses, her filters); validated on her first 10M words: 99.5% of rows identical, cause object mean 3.97 vs her 3.96. "
         "Only verbs with at least 30 scored instances are shown. Bars = 95% CI.\n"
         "(S) sufficiency verb, (N) necessity/enabling verb.  * object is often the person made/allowed to act, not the effect (and pronouns are scored, as in her code).  "
         "Dashed line = neutral (5).\n"
         "Not in panel A because the direct object is not the effect: depend, arise, stem, result, lead, contribute.  Removed by her frequency filter (her lemma count < 2 per million): "
         "compel, facilitate, hinder, inhibit, necessitate, provoke.",
         fontsize=7, color=MUTED, va="bottom")
fig.tight_layout(rect=(0, 0.06, 1, 0.97))
fig.savefig(config.FIGURES / "glass_pipeline_month.png", dpi=200); fig.savefig(config.FIGURES / "glass_pipeline_month.pdf"); print("ok")
