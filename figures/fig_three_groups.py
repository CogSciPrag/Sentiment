"""Bar chart of selected expressions in three groups, all four corpora side by side (3 Nov 2026, Dean's request).

Group 1: cause(s) of and cause (v.); Group 2: clearly sufficiency-entailing (ensure, force, guarantee, make, sufficient for);
Group 3: clearly not sufficiency-entailing (allow, encourage, let, make a difference to, necessary for, require).
One bar per corpus (mean effect score of the expression in that corpus, combined N+V, all forms); a short black dash on each bar =
that corpus's baseline of all other verbs; hatched bar = not significantly different from that baseline (Welch t-test,
Holm-corrected within scoring x corpus x measure x polarity x baseline type). Bars start at the bottom of the axis (1).
'cause(s) of' is the item 'the cause of' (every 'cause of' / 'causes of' + noun: also 'a cause of', 'main cause of', 'cause of death').
Input: data/results/pvalues_all.tsv (step 9). Output: data/figures/fig_three_groups_<scoring>.png and .pdf
Usage: python figures/fig_three_groups.py [scoring]     (raw, band, adj or war; default raw)
"""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from causesent import config
import causesent.style  # Helvetica

sc = sys.argv[1] if len(sys.argv) > 1 else "raw"
config.ensure_dirs(); out = str(config.FIGURES / f"fig_three_groups_{sc}.png")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
LABEL = {"raw": "raw scores", "band": "neutral band (4–6) removed", "adj": "adjectives and noun compounds averaged into the noun",
         "war": "Warriner et al. (2013) valence norms"}
SCALE = "Warriner valence" if sc == "war" else "Hedonometer score"
CORP = [("ententen21", "enTenTen21 (web, 52B words)", "#2b6cd4"), ("bnc", "BNC (British, 100M words)", "#f2b705"),
        ("coca", "COCA (American, 1B words)", "#1a9e8f"), ("reddit", "AskReddit Jan 2018 (139M words)", "#8e3fb5")]
GROUPS = [("Group 1", [("the cause of", "cause(s) of"), ("cause", "cause (v.)")]),
          ("Group 2 – Clearly sufficiency-entailing", [("ensure", "ensure"), ("force", "force"), ("guarantee", "guarantee"),
                                                       ("make", "make *"), ("sufficient for", "sufficient\nfor")]),
          ("Group 3 – Clearly not sufficiency-entailing", [("allow", "allow"), ("encourage", "encourage"), ("let", "let"),
                                                           ("make a difference to", "make a\ndifference to"),
                                                           ("necessary for", "necessary\nfor"), ("require", "require")])]
MIN_N = 30

P = pd.read_csv(config.RESULTS / "pvalues_all.tsv", sep="\t")
P = P[(P.scoring == sc) & (P.measure == "NV") & (P.polarity == "combined") & (P.baseline_type == "verbs")]
def get(item, corp):
    r = P[(P.item == item) & (P.corpus == corp)]
    return None if r.empty or r.n.iloc[0] < MIN_N else r.iloc[0]

BW, GAP = 0.2, 0.9                     # bar width; word clusters are 1 apart, groups are separated by an extra GAP
xs, labels, bounds = [], [], []        # xs: (x of the cluster, item); bounds: (first, last) index of each group in xs
x0 = 0.0
for gname, words in GROUPS:
    first = len(xs)
    for item, lb in words:
        xs.append((x0, item)); labels.append(lb); x0 += 1.0
    bounds.append((first, len(xs) - 1)); x0 += GAP
fig, ax = plt.subplots(figsize=(15, 7.2))
off = {c: (k - 1.5) * BW * 1.08 for k, (c, _, _) in enumerate(CORP)}
for xc, item in xs:
    for c, name, fc in CORP:
        r = get(item, c); xb = xc + off[c]
        if r is None:
            ax.text(xb, 1.1, "n<30", rotation=90, fontsize=6.5, color=MUTED, ha="center", va="bottom"); continue
        sig = r.p_holm < 0.05
        ax.bar(xb, r["mean"] - 1, bottom=1, width=BW, color=fc if sig else "white", edgecolor=INK if sig else fc,
               linewidth=0.5 if sig else 1.0, hatch=None if sig else "////", zorder=2)
        ax.plot([xb - BW / 2, xb + BW / 2], [r.baseline] * 2, color=INK, lw=1.6, zorder=4, solid_capstyle="butt")
        yl = r.baseline + 0.06 if r["mean"] < r.baseline < r["mean"] + 0.45 else r["mean"] + 0.06   # keep the number clear of the baseline dash
        ax.text(xb, yl, f"{r['mean']:.2f}", rotation=90, fontsize=6.6, color=INK, ha="center", va="bottom", zorder=5)
ax.axhline(5, color=MUTED, lw=0.7, ls=":", zorder=1)
ax.set_xticks([xc for xc, _ in xs]); ax.set_xticklabels(labels, rotation=0, fontsize=9, fontstyle="italic")
# group headings and separators
for gi, ((gname, words), (i0, i1)) in enumerate(zip(GROUPS, bounds)):
    ax.text((xs[i0][0] + xs[i1][0]) / 2, 7.55, gname, ha="center", va="center", fontsize=10.5, color=INK)
    if gi: ax.axvline((xs[i0 - 1][0] + xs[i0][0]) / 2, color=MUTED, lw=0.8, ls="--", zorder=1)
ax.set_xlim(xs[0][0] - 0.7, xs[-1][0] + 0.7); ax.set_ylim(1, 7.8); ax.set_yticks(range(1, 8))
ax.set_ylabel(f"Mean {SCALE} of the effect\n(aggregate over all frames)", fontsize=9, color=MUTED)
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.tick_params(colors=MUTED, labelsize=8.5); ax.tick_params(axis="x", labelcolor=INK)
h = [Patch(facecolor=fc, edgecolor=INK, linewidth=0.5, label=name) for _, name, fc in CORP]
h.append(Patch(facecolor="white", edgecolor=MUTED, hatch="////", label="hatched = not significantly different from that corpus's baseline (Holm)"))
h.append(Line2D([], [], color=INK, lw=1.6, label="black dash = that corpus's baseline (all other verbs)"))
fig.legend(handles=h, loc="lower center", ncol=3, fontsize=8.3, frameon=False, bbox_to_anchor=(0.5, 0.085))
ax.set_title(f"Effect sentiment of selected expressions in four corpora ({LABEL[sc]})", fontsize=12, loc="left", pad=22)
fig.text(0.01, 0.012,
         "Bar = mean effect score of the expression in that corpus (combined noun + verb, positive and negative forms; number above each bar); bars start at 1; dotted line = neutral (5). "
         "Only means with ≥ 30 scored lines.\ncause(s) of = the item 'the cause of': every 'cause of' / 'causes of' + noun ('a cause of', 'the main cause of', 'cause of death'); cause (v.) = the verb. "
         "* make: causative frames only (make N ADJ / N V / N V-ed).\nSignificance: Welch t-test against that corpus's baseline of all other verbs, Holm-corrected within corpus and measure, p < .05.",
         fontsize=7.2, color=MUTED, va="bottom")
fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.27)
fig.savefig(out, dpi=200); fig.savefig(out.replace(".png", ".pdf")); print("saved", out)
