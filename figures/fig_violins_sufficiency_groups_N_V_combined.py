"""Only-N and only-V violins of the 20 expressions in ONE figure (5 Nov 2026, Dean): top panel = only the effect noun scored
(measure N, score_N), bottom panel = only the effect verb scored (measure V, score_V). No group titles above the violins (Dean);
the three groups (cause and because | clearly sufficiency-entailing | clearly not) are still separated by dotted lines.
Same violins, markers, baselines and Holm tests as fig_violins_sufficiency_groups_N_V.py, each panel with its own measure's
baselines and grey band. The same expression order in both panels (within each group by the only-N mean) so the panels line up.
Cells with < 30 lines are left out (no marker; violin and mean over the remaining corpora, equal weights; note under the label).
Usage: python figures/fig_violins_sufficiency_groups_N_V_combined.py [scoring]
Output: data/figures/fig_violins_sufficiency_groups_onlyN_onlyV_<scoring>.png/.pdf + .tsv
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

sc = sys.argv[1] if len(sys.argv) > 1 else "raw"
config.ensure_dirs(); out = str(config.FIGURES / f"fig_violins_sufficiency_groups_onlyN_onlyV_{sc}.png")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
LABEL = {"raw": "raw scores", "band": "neutral band (4–6) removed", "adj": "adjectives and noun compounds averaged into the noun", "war": "Warriner et al. (2013) valence norms"}
SCALE = "Warriner valence" if sc == "war" else "Hedonometer score"
PANEL = {"N": "Only nouns scored", "V": "Only verbs scored"}
COL = {"N": "score_N", "V": "score_V"}
GROUPS = {"Cause and because": ["cause", "because (clause)"],
          "Clearly sufficiency-entailing": ["compel", "enough for", "ensure", "force", "guarantee", "make", "oblige", "produce", "sufficient for"],
          "Clearly not sufficiency-entailing": ["allow", "contribute to", "enable", "encourage", "influence", "let", "make a difference to", "necessary for", "permit"]}
ITEMS = [i for g in GROUPS.values() for i in g]; MIN_N = 30
MK = {"ententen21": ("o", "enTenTen21 (web, 52B words)", "#2b6cd4"), "bnc": ("s", "BNC (British, 100M words)", "#f2b705"),
      "reddit": ("D", "AskReddit Jan 2018 (139M words)", "#8e3fb5"), "coca": ("^", "COCA (American, 1B words)", "#1a9e8f")}
SHORT = {"ententen21": "enTenTen21", "bnc": "BNC", "reddit": "AskReddit", "coca": "COCA"}
off = {"bnc": -0.28, "ententen21": -0.095, "reddit": 0.095, "coca": 0.28}
DISPLAY = {**DISPLAY, "because (clause)": "because"}
lab = lambda it: DISPLAY.get(it, it)
BREAK = {"sufficient for": "sufficient\nfor", "enough for": "enough\nfor", "contribute to": "contribute\nto", "necessary for": "necessary\nfor", "make a difference to": "make a\ndifference\nto"}

PALL = pd.read_csv(config.RESULTS / "pvalues_all.tsv", sep="\t")
PALL = PALL[(PALL.scoring == sc) & (PALL.polarity == "combined") & (PALL.baseline_type == "verbs")]
L = pd.read_pickle(config.RESULTS / f"lines_{sc}.pkl"); L = L[L.item.isin(ITEMS)]
D = {}
for M in "NV":
    P = PALL[PALL.measure == M]; Q = P[(P.n >= MIN_N) & P.item.isin(ITEMS)]
    W = Q.pivot(index="item", columns="corpus", values="mean"); S = Q.pivot(index="item", columns="corpus", values="p_holm")
    LM = L[L[COL[M]].notna()]
    D[M] = dict(P=P, Q=Q, W=W, S=S, mean4=W.mean(axis=1), G={k: g[COL[M]].values for k, g in LM.groupby(["corpus", "item"])})

items = []; x = []; gx = {}; pos = 0.0          # one order for both panels: by the only-N mean within each group
for gname, its in GROUPS.items():
    its = sorted(its, key=lambda i: D["N"]["mean4"][i]); start = pos
    for i in its: items.append(i); x.append(pos); pos += 1
    gx[gname] = (start, pos - 1); pos += 0.9
x = np.array(x)

fig, axes = plt.subplots(2, 1, figsize=(24, 15))
HALF = 0.42
for ax, M in zip(axes, "NV"):
    P, W, S, mean4, G = (D[M][k] for k in ("P", "W", "S", "mean4", "G"))
    LO, HI = bounds(sc, M); col = band_colour_fn(LO, HI); NC = {}
    for xi, it in zip(x, items):
        cs = [c for c in MK if len(G.get((c, it), [])) >= MIN_N]; assert len(cs) >= 1, (M, it, cs); NC[it] = len(cs)
        data = np.concatenate([G[(c, it)] for c in cs]); w = np.concatenate([np.full(len(G[(c, it)]), 1 / len(cs) / len(G[(c, it)])) for c in cs])
        wmean = np.average(data, weights=w); assert abs(wmean - mean4[it]) < 1e-4, (M, it, wmean, mean4[it])
        kde = gaussian_kde(data, weights=w, bw_method=0.25)
        g = np.linspace(data.min(), data.max(), 300); d = kde(g); d = d / d.max() * HALF
        ax.fill_betweenx(g, xi - d, xi + d, facecolor=col(wmean), edgecolor=INK, linewidth=0.5, zorder=2)
        ax.hlines(wmean, xi - HALF * 0.9, xi + HALF * 0.9, color=INK, lw=1.6, zorder=3)
    bls = {}
    for c, (mk, name, fc) in MK.items():
        y = W[c].reindex(items).values; ok = ~np.isnan(y); sig = ((S[c].reindex(items) < 0.05).values) & ok; ns = ok & ~sig
        ax.scatter((x + off[c])[sig], y[sig], marker=mk, s=30, facecolor=fc, edgecolor=INK, linewidths=0.6, zorder=4, label=name)
        ax.scatter((x + off[c])[ns], y[ns], marker=mk, s=30, facecolor="white", edgecolor=fc, linewidths=1.3, zorder=4)
        med = np.array([np.median(G[(c, i)]) if len(G.get((c, i), [])) >= MIN_N else np.nan for i in items])
        ax.hlines(med[ok], (x + off[c] - 0.08)[ok], (x + off[c] + 0.08)[ok], color=fc, lw=2.2, zorder=3)
        bl = P[P.corpus == c].baseline.dropna().iloc[0]; bls[c] = bl
        ax.axhline(bl, color=fc, lw=1.0, ls="--", zorder=1, label=f"{SHORT[c]} baseline (all other verbs, dashed)")
    ax.axhline(5, color=MUTED, lw=0.7, ls=":", zorder=0)
    ax.set_xticks(x); ax.set_xticklabels([BREAK.get(i, lab(i)) for i in items], fontsize=14.5, fontstyle="italic")
    for xi, it in zip(x, items):
        if NC[it] < 4: ax.text(xi, -0.2, f"{NC[it]} of 4 corpora\nwith n ≥ 30", ha="center", va="top", fontsize=10, color=MUTED, transform=matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes))
    _g = list(gx.values())
    for (_, e1), (s2, _) in zip(_g[:-1], _g[1:]): ax.vlines((e1 + s2) / 2, 1, 9, color=MUTED, lw=0.8, ls=(0, (2, 3)), zorder=0)
    ax.text(-0.5, 9.35, PANEL[M], ha="left", va="bottom", fontsize=16, color=INK)           # panel label (Dean, 5 Nov 2026)
    ax.text(x.max() + 0.5, 9.35, "baselines (all other verbs, scored the same way): " + ", ".join(f"{SHORT[c]} {bls[c]:.2f}" for c in MK),
            ha="right", va="bottom", fontsize=10.5, color=MUTED)
    ax.set_ylim(1, 10.0); ax.set_yticks(range(1, 10)); ax.set_xlim(-0.65, x.max() + 0.65)
    ax.set_ylabel(f"{SCALE} of the effect (per line)", fontsize=13, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=13); ax.tick_params(axis="x", labelsize=14.5, labelcolor=INK)
    D[M]["NC"] = NC

h, l = axes[0].get_legend_handles_labels()
h.append(Line2D([], [], marker="o", color="none", markerfacecolor="white", markeredgecolor=MUTED)); l.append("hollow marker = not significantly different from that corpus's baseline (Holm)")
h.append(Line2D([], [], color=INK, lw=1.6)); l.append("mean of the corpora (each weighted equally)")
h.append(Line2D([], [], color=MUTED, lw=2.2)); l.append("short bar in the corpus colour = that corpus's median")
fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.055, 0.955), fontsize=9, frameon=False, ncol=3, columnspacing=2.0)
fig.text(0.06, 0.985, f"Effect sentiment of 20 expressions, per line ({LABEL[sc]}; only the effect noun vs only the effect verb scored; violin = four corpora weighted equally, markers = each corpus)", fontsize=12.5, ha="left", va="top")
fig.subplots_adjust(left=0.06, right=0.995, top=0.88, bottom=0.11, hspace=0.42)
fig.savefig(out, dpi=200); fig.savefig(out.replace(".png", ".pdf"))
T = pd.concat([D[M]["Q"].assign(measure=M) for M in "NV"])
T.assign(group=T.item.map({i: g for g, its in GROUPS.items() for i in its}), scored=T.measure.map({"N": "only nouns", "V": "only verbs"})).sort_values(["measure", "group", "item", "corpus"])[["scored", "group", "item", "corpus", "n", "mean", "baseline", "diff", "p_holm", "stars"]].to_csv(out.replace(".png", ".tsv"), sep="\t", index=False, float_format="%.5g")
print("saved", out)
