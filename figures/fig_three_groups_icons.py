"""Same three groups as fig_three_groups.py, but ONE bar per expression (mean of the four corpora's means, coloured red-green as in
the full results chart) with one small marker per corpus at that corpus's own mean: filled = significantly different from that corpus's baseline, hollow = not (3 Nov 2026, Dean's request).
Older description of the sibling figure:

Group 1: cause(s) of and cause (v.); Group 2: clearly sufficiency-entailing (ensure, force, guarantee, make, sufficient for);
Group 3: clearly not sufficiency-entailing (allow, encourage, let, make a difference to, necessary for, require).
One bar per corpus (mean effect score of the expression in that corpus, combined N+V, all forms); a short black dash on each bar =
that corpus's baseline of all other verbs; hatched bar = not significantly different from that baseline (Welch t-test,
Holm-corrected within scoring x corpus x measure x polarity x baseline type). Bars start at the bottom of the axis (1).
'cause(s) of' is the item 'the cause of' (every 'cause of' / 'causes of' + noun: also 'a cause of', 'main cause of', 'cause of death').
Input: data/results/pvalues_all.tsv (step 9). Output: data/figures/fig_three_groups_icons_<scoring>.png and .pdf
Usage: python figures/fig_three_groups_icons.py [scoring]     (raw, band, adj or war; default raw)
"""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from causesent import config
import causesent.style  # Helvetica

sc = sys.argv[1] if len(sys.argv) > 1 else "raw"
config.ensure_dirs(); out = str(config.FIGURES / f"fig_three_groups_icons_{sc}.png")
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
CMAP = LinearSegmentedColormap.from_list("rg", ["#b2182b", "#e8816b", "#dcdad4", "#7fbf7b", "#1b7837"])
from causesent.palette import centre, bounds, band_colour_fn
CENTRE = centre(sc); LO, HI = bounds(sc); col = band_colour_fn(LO, HI)   # grey band = from the lowest to the highest of the four corpora's baselines (3 Nov 2026)
MARK = {"ententen21": "o", "bnc": "s", "coca": "^", "reddit": "D"}
OFF = {"bnc": -0.27, "ententen21": -0.09, "reddit": 0.09, "coca": 0.27}     # as in the full results chart

P = pd.read_csv(config.RESULTS / "pvalues_all.tsv", sep="\t")
P = P[(P.scoring == sc) & (P.measure == "NV") & (P.polarity == "combined") & (P.baseline_type == "verbs")]
def get(item, corp):
    r = P[(P.item == item) & (P.corpus == corp)]
    return None if r.empty or r.n.iloc[0] < MIN_N else r.iloc[0]

xs, labels, bounds = [], [], []
x0 = 0.0
for gname, words in GROUPS:
    first = len(xs)
    for item, lb in words:
        xs.append((x0, item)); labels.append(lb); x0 += 1.0
    bounds.append((first, len(xs) - 1)); x0 += 0.9
fig, ax = plt.subplots(figsize=(15, 7.2))
for xc, item in xs:
    rows = {c: get(item, c) for c, _, _ in CORP}
    means = [r["mean"] for r in rows.values() if r is not None]
    m = float(np.mean(means))
    ax.bar(xc, m - 1, bottom=1, width=0.78, color=col(m), edgecolor=INK, linewidth=0.4, zorder=2)
    for c, name, fc in CORP:
        r = rows[c]
        if r is None: continue
        sig = r.p_holm < 0.05
        ax.scatter(xc + OFF[c], r["mean"], marker=MARK[c], s=46, facecolor=fc if sig else "white", edgecolor=INK if sig else fc,
                   linewidths=0.7 if sig else 1.3, zorder=4)
for c, name, fc in CORP:
    bl = P[P.corpus == c].baseline.dropna().iloc[0]
    ax.axhline(bl, color=fc, lw=1.0, ls="--", zorder=3)
ax.axhline(5, color=MUTED, lw=0.7, ls=":", zorder=1)
ax.set_xticks([xc for xc, _ in xs]); ax.set_xticklabels(labels, rotation=0, fontsize=9, fontstyle="italic")
for gi, ((gname, words), (i0, i1)) in enumerate(zip(GROUPS, bounds)):
    ax.text((xs[i0][0] + xs[i1][0]) / 2, 7.55, gname, ha="center", va="center", fontsize=10.5, color=INK)
    if gi: ax.axvline((xs[i0 - 1][0] + xs[i0][0]) / 2, color=MUTED, lw=0.8, ls="--", zorder=1)
ax.set_xlim(xs[0][0] - 0.7, xs[-1][0] + 0.7); ax.set_ylim(1, 7.8); ax.set_yticks(range(1, 8))
ax.set_ylabel(f"Mean {SCALE} of the effect\n(aggregate over all frames)", fontsize=9, color=MUTED)
ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.tick_params(colors=MUTED, labelsize=8.5); ax.tick_params(axis="x", labelcolor=INK)
h = [Line2D([], [], marker=MARK[c], color="none", markerfacecolor=fc, markeredgecolor=INK, markersize=7, label=name) for c, name, fc in CORP]
h.append(Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor=MUTED, markersize=7, markeredgewidth=1.3,
                label="hollow marker = not significantly different from that corpus's baseline (Holm)"))
for c, name, fc in CORP:
    h.append(Line2D([], [], color=fc, lw=1.2, ls="--", label=f"{name.split(' (')[0]} baseline, all other verbs: {P[P.corpus == c].baseline.dropna().iloc[0]:.2f}"))
fig.legend(handles=h, loc="lower center", ncol=3, fontsize=8.3, frameon=False, bbox_to_anchor=(0.5, 0.075))
ax.set_title(f"Effect sentiment of selected expressions ({LABEL[sc]}; bar = mean of the four corpora, markers = each corpus)", fontsize=12, loc="left", pad=22)
fig.text(0.01, 0.012,
         f"Colour: grey = between the lowest and highest of the four corpora's baselines ({LO:.2f}–{HI:.2f}), fading to red below and to green above; bars start at 1; dotted line = 5. Only means with ≥ 30 scored lines. Filled marker = significantly different from that corpus's baseline "
         "(Welch t-test, Holm-corrected, p < .05).\ncause(s) of = the item 'the cause of': every 'cause of' / 'causes of' + noun ('a cause of', 'the main cause of', 'cause of death'); cause (v.) = the verb. "
         "* make: causative frames only (make N ADJ / N V / N V-ed).\nCombined noun + verb effect score, positive and negative forms; baselines = every other lexical verb, same rules.",
         fontsize=7.2, color=MUTED, va="bottom")
fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.27)
fig.savefig(out, dpi=200); fig.savefig(out.replace(".png", ".pdf")); print("saved", out)
