"""Pairwise matrix of the 20 expressions in fig_violins_sufficiency_groups (10 Nov 2026, Dean).

Lower triangle: one cell per pair, value = column minus row (changed from row minus column, 10 Nov 2026, Dean) (four-corpus mean, each corpus 25%). Each cell shows
the 95% CI of the difference (bootstrap over effect words within each corpus, 2,000 resamples; words shared by two
expressions are resampled together) and the Holm-corrected p from the same bootstrap (difference / bootstrap SE,
normal approximation; Holm across all 190 pairs), so p and CI always agree. The .tsv also has the across-corpora
paired t-test (df = 3) for reference: with only four corpora and 190 pairs it has almost no power after Holm. Colour = the difference (green = column more positive, red = column more negative, clipped at +/-1; colours from causesent.palette).
Bold = Holm p < .05. Diagonal: the expression's own mean.
Input: data/results/lines_raw.pkl. Output: data/figures/fig_pairwise_matrix_raw.{pdf,png,tsv} (column minus row) and fig_pairwise_matrix_raw_upper.{pdf,png} (row minus column)
Usage: python figures/fig_pairwise_matrix.py
"""
import itertools
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy import stats
from statsmodels.stats.multitest import multipletests
from causesent import config
import causesent.style  # Helvetica

GROUPS = {"Cause and because": ["cause", "because (clause)"],
          "Clearly sufficiency-entailing": ["compel", "enough for", "ensure", "force", "guarantee", "make", "oblige", "produce", "sufficient for"],
          "Clearly not sufficiency-entailing": ["allow", "contribute to", "enable", "encourage", "influence", "let", "make a difference to", "necessary for", "permit"]}
CORPORA = ["ententen21", "bnc", "reddit", "coca"]; B = 2000; rng = np.random.default_rng(20261110)
INK, MUTED = "#0b0b0b", "#52514e"
CELL_P, CELL_CI = 11, 8.1    # text inside the cells, larger (10 Nov 2026, Dean)
config.ensure_dirs(); OUT = config.FIGURES / "fig_pairwise_matrix_raw"

ALL = [i for g in GROUPS.values() for i in g]
L = pd.read_pickle(config.RESULTS / "lines_raw.pkl")
L = L.loc[L.agg_score.notna() & L.item.isin(ALL), ["corpus", "item", "agg_score", "N_keys", "V_keys"]].copy()
L["key"] = L.N_keys.fillna("") + "|" + L.V_keys.fillna("")
per = L.groupby(["item", "corpus"]).agg_score.mean().unstack()[CORPORA]; four = per.mean(1)
items = [i for g in GROUPS.values() for i in sorted(g, key=lambda x: four[x])]     # same order as the violin figure
per = per.loc[items]; four = four.loc[items]; k = len(items)

S, N = {}, {}
for c in CORPORA:
    A = L[L.corpus == c].groupby(["key", "item"]).agg_score.agg(["sum", "size"]).unstack(fill_value=0)
    S[c] = A["sum"].reindex(columns=items, fill_value=0).values; N[c] = A["size"].reindex(columns=items, fill_value=0).values
boot = np.zeros((B, k))
for b in range(B):
    ms = []
    for c in CORPORA:
        idx = rng.integers(0, len(S[c]), len(S[c])); ms.append(S[c][idx].sum(0) / N[c][idx].sum(0))
    boot[b] = np.mean(ms, axis=0)

rows = []
for i, j in itertools.combinations(range(k), 2):          # i < j; the cell (row j, column i) shows column - row
    r, c_ = items[j], items[i]; d = per.loc[c_] - per.loc[r]
    bd = boot[:, i] - boot[:, j]; lo, hi = np.percentile(bd, [2.5, 97.5]); diff = four[c_] - four[r]
    rows.append(dict(row=r, col=c_, diff=diff, ci_lo=lo, ci_hi=hi, boot_se=bd.std(ddof=1),
                     p=2 * stats.norm.sf(abs(diff) / bd.std(ddof=1)),                      # same bootstrap as the CI
                     p_corpus_t3=stats.ttest_1samp(d.values, 0).pvalue))                   # across-corpora test, for reference only
R = pd.DataFrame(rows); R["p_holm"] = multipletests(R.p, method="holm")[1]; R["p_corpus_t3_holm"] = multipletests(R.p_corpus_t3, method="holm")[1]
R.to_csv(str(OUT) + ".tsv", sep="\t", index=False, float_format="%.5g")

lab = lambda it: {"because (clause)": "because"}.get(it, it)
fmt_p = lambda p: "p < .001" if p < 0.001 else ("p = 1" if p > 0.995 else f"p = {p:.3f}".replace("0.", ".") if p < 0.01 else f"p = {p:.2f}".replace("0.", "."))
fmt = lambda x: f"{x + 0.0:.2f}".replace("-0.00", "0.00").replace("-", "−")   # no + sign, so the larger text fits the cell
from causesent.palette import CMAP as cmap   # same red-grey-green as the violin figures (10 Nov 2026, Dean)
norm = Normalize(-1, 1)

def draw(upper=False):
    """upper=False: lower triangle, column minus row.  upper=True: the same matrix flipped on the diagonal (upper
    triangle), row minus column -- every pair keeps the same value, p and CI. No group dividers; all borders equal (Dean, 10 Nov 2026)."""
    fig, ax = plt.subplots(figsize=(19, 17))
    for _, rr in R.iterrows():
        y, x = items.index(rr.row), items.index(rr.col)
        if upper: x, y = y, x
        v = rr["diff"]; col = cmap(norm(np.clip(v, -1, 1)))
        ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=col, edgecolor="white", lw=1.2))
        tc = "white" if abs(v) > 0.75 else INK; bold = "bold" if rr.p_holm < 0.05 else "normal"
        ax.text(x, y - 0.19, fmt_p(rr.p_holm), ha="center", va="center", fontsize=CELL_P, color=tc, fontweight=bold)
        ax.text(x, y + 0.2, f"[{fmt(rr.ci_lo)},\u2009{fmt(rr.ci_hi)}]", ha="center", va="center", fontsize=CELL_CI, color=tc, fontweight=bold)
    for d, it in enumerate(items):                               # diagonal: the expression's mean
        ax.add_patch(plt.Rectangle((d - 0.5, d - 0.5), 1, 1, facecolor="white", edgecolor=MUTED, lw=1.2))
        ax.text(d, d, f"{four[it]:.2f}", ha="center", va="center", fontsize=10, color=INK)
    ax.set_xlim(-0.5, k - 0.5); ax.set_ylim(k - 0.5, -0.5); ax.set_aspect("equal")
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    if upper:
        ax.xaxis.tick_top(); ax.yaxis.tick_right()
        ax.set_xticklabels([lab(i) for i in items], fontsize=15, fontstyle="italic", rotation=40, ha="left", rotation_mode="anchor")
        ax.set_yticklabels([lab(i) for i in items], fontsize=15, fontstyle="italic")
    else:
        ax.set_xticklabels([lab(i) for i in items], fontsize=15, fontstyle="italic", rotation=40, ha="right", rotation_mode="anchor")
        ax.set_yticklabels([lab(i) for i in items], fontsize=15, fontstyle="italic")
    for s_ in ax.spines.values(): s_.set_visible(False)
    ax.tick_params(length=0, pad=6)
    if upper: fig.subplots_adjust(left=0.01, right=0.88, top=0.89, bottom=0.02); cx, cy = 0.06, 0.25
    else:     fig.subplots_adjust(left=0.12, right=0.99, top=0.98, bottom=0.11); cx, cy = 0.70, 0.80
    cax = fig.add_axes([cx, cy, 0.22, 0.018]); cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks([-1, -0.5, 0, 0.5, 1]); cb.set_ticklabels(["\u2264\u22121", "\u22120.5", "0", "+0.5", "\u2265+1"]); cb.ax.tick_params(labelsize=11, colors=MUTED); cb.outline.set_visible(False)
    cax.set_title(f"Difference, {'row minus column' if upper else 'column minus row'} (Hedonometer points)", fontsize=12.5, color=INK, pad=8)
    fig.text(cx, cy - 0.035, "Each cell: Holm-corrected p (190 pairs) and 95% CI,\nboth from a bootstrap over effect words\nBold = Holm p < .05   Diagonal = the expression's mean",
             fontsize=12, color=INK, ha="left", va="top", linespacing=1.5)
    fig.canvas.draw(); r_ = fig.canvas.get_renderer(); cw = ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0]
    print("widest cell text / cell width:", round(max(t.get_window_extent(r_).width for t in ax.texts) / cw, 3))
    out = str(OUT) + ("_upper" if upper else "")
    fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf"); plt.close(fig); print("saved", out)

draw(upper=False)   # fig_pairwise_matrix_raw: lower triangle, column minus row
draw(upper=True)    # fig_pairwise_matrix_raw_upper: flipped on the diagonal, row minus column
print((R.p_holm < 0.05).sum(), "of", len(R), "pairs Holm p < .05")
print(R[(R.p_holm < 0.05) & ~R.row.eq("cause") & ~R.col.eq("cause")].round(3).to_string(index=False))
