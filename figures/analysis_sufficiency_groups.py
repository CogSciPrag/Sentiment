"""Statistical comparison of the two sufficiency groups (10 Nov 2026, Dean):
clearly sufficiency-entailing (force ... guarantee, 9) vs clearly not sufficiency-entailing (contribute to ... make a difference to, 9).

The unit of inference is the EXPRESSION, not the line: the claim is about kinds of expressions, there are only 9 + 9 of them,
and lines are not independent (they share effect words, and the score is a lexicon lookup on those words). With ~423k lines,
any line-level test is 'significant' for trivial differences, so line-level models are reported only as a check.
Every expression's score = mean of its four corpus means (each corpus 25%), exactly the black bar in
fig_violins_sufficiency_groups.

  1. Item-level test: difference of group means, Welch t, Hedges' g (bootstrap CI), exact permutation test over all
     C(18, 9) = 48,620 relabellings, exact Mann-Whitney, JZS Bayes factor (r = 0.707), TOST equivalence test (+/- DELTA).
  2. Mixed models: (a) the 72 expression x corpus means, y ~ group + corpus + (1 | expression), plus a likelihood-ratio
     test of group x corpus; (b) the lines, agg_score ~ group + corpus + (1 | expression) + (1 | expression:corpus).
  3. Beyond the mean: share of negative effects (< 5), clearly negative (< 4), positive (> 6), SD, 10th/90th percentiles,
     and the distance between the two groups' whole distributions (Wasserstein-1 and Kolmogorov-Smirnov), each tested
     with the same exact item-level permutation; Holm across the distribution measures.
  4. Robustness: neutral band removed / Warriner norms, each corpus on its own, without make, without each expression in turn.
  5. Sensitivity: the smallest difference this design (9 vs 9 expressions) detects with 80% power.
Input: data/results/lines_{raw,band,war}.pkl, pvalues_all.tsv. Output: data/figures/analysis_sufficiency_groups.{pdf,png,tsv,txt}
Usage: python figures/analysis_sufficiency_groups.py
"""
import itertools, io, contextlib
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats, integrate
import statsmodels.formula.api as smf
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.multitest import multipletests
from causesent import config
import causesent.style  # Helvetica

SUF = ["force", "compel", "make", "oblige", "sufficient for", "enough for", "produce", "ensure", "guarantee"]
NOT = ["contribute to", "permit", "let", "allow", "influence", "enable", "encourage", "necessary for", "make a difference to"]
ITEMS = SUF + NOT; CORPORA = ["ententen21", "bnc", "reddit", "coca"]
DELTA = 0.20          # equivalence bound, Hedonometer points (corpus baselines span 0.09; cause sits 1.5 below them)
GRID = np.round(np.arange(1.0, 9.0001, 0.01), 2)
rng = np.random.default_rng(20261110)
config.ensure_dirs(); OUT = config.FIGURES / "analysis_sufficiency_groups"
INK, MUTED, GRIDC = "#0b0b0b", "#52514e", "#e6e5e0"
CC = {"ententen21": "#2b6cd4", "bnc": "#f2b705", "reddit": "#8e3fb5", "coca": "#1a9e8f"}
GC = {"suf": "#5f8f5a", "not": "#8a6a9e"}   # group colours (green-ish / purple-ish), text stays in ink
log = io.StringIO(); say = lambda *a: print(*a, file=log)


def load(sc, items=ITEMS + ["cause"]):
    L = pd.read_pickle(config.RESULTS / f"lines_{sc}.pkl")
    L = L.loc[L.agg_score.notna() & L.item.isin(items), ["corpus", "item", "agg_score"]].copy()
    L["w"] = 0.25 / L.groupby(["item", "corpus"]).agg_score.transform("size")     # each corpus 25% within an expression
    return L


def item_table(L):
    """Per expression: 4-corpus mean and distribution measures (all with the 25% corpus weights)."""
    rows = []
    for it, g in L.groupby("item"):
        s, w = g.agg_score.values, g.w.values; o = np.argsort(s); cw = np.cumsum(w[o])
        q = lambda p: s[o][np.searchsorted(cw, p)]
        m = np.average(s, weights=w)
        rows.append(dict(item=it, mean=m, p_neg=w[s < 5].sum(), p_clear_neg=w[s < 4].sum(), p_pos=w[s > 6].sum(),
                         sd=np.sqrt(np.average((s - m) ** 2, weights=w)), q10=q(0.10), q90=q(0.90)))
    return pd.DataFrame(rows).set_index("item")


def cdf_matrix(L, items):
    """Weighted CDF of each expression on a 0.01 grid (rows = expressions)."""
    M = np.zeros((len(items), len(GRID)))
    for k, it in enumerate(items):
        g = L[L.item == it]; h = np.bincount(np.searchsorted(GRID, np.round(g.agg_score.values, 2)), weights=g.w.values, minlength=len(GRID))
        M[k] = np.cumsum(h)
    return M


SPLITS = np.array([[i in c for i in range(18)] for c in itertools.combinations(range(18), 9)])   # 48,620 x 18, True = 'suf'
def perm_p(v):
    """Exact two-sided permutation p for difference of group means; v ordered as ITEMS (first 9 = suf)."""
    v = np.asarray(v, float); d = (SPLITS @ v) / 9 - (~SPLITS @ v) / 9; obs = v[:9].mean() - v[9:].mean()
    return obs, np.mean(np.abs(d) >= abs(obs) - 1e-12)


def hedges_g(a, b):
    n1, n2 = len(a), len(b); sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / sp * (1 - 3 / (4 * (n1 + n2) - 9))


def jzs_bf10(t, n1, n2, r=0.707):
    """Two-sample JZS Bayes factor (Rouder et al. 2009), Cauchy prior with scale r on delta."""
    N = n1 * n2 / (n1 + n2); nu = n1 + n2 - 2
    f = lambda g: (1 + N * g * r ** 2) ** -0.5 * (1 + t ** 2 / ((1 + N * g * r ** 2) * nu)) ** (-(nu + 1) / 2) * (2 * np.pi) ** -0.5 * g ** -1.5 * np.exp(-1 / (2 * g))
    return integrate.quad(f, 0, np.inf)[0] / (1 + t ** 2 / nu) ** (-(nu + 1) / 2)


def item_tests(v, label):
    a, b = v[:9], v[9:]
    obs, pp = perm_p(v); w = stats.ttest_ind(a, b, equal_var=False); mw = stats.mannwhitneyu(a, b, method="exact")
    se = np.sqrt(a.var(ddof=1) / 9 + b.var(ddof=1) / 9); df = w.df
    ci95 = obs + np.array([-1, 1]) * stats.t.ppf(0.975, df) * se; ci90 = obs + np.array([-1, 1]) * stats.t.ppf(0.95, df) * se
    p_tost = max(stats.t.sf((obs + DELTA) / se, df), stats.t.cdf((obs - DELTA) / se, df))
    boot = [hedges_g(rng.choice(a, 9), rng.choice(b, 9)) for _ in range(5000)]
    return dict(analysis=label, suf=a.mean(), not_=b.mean(), diff=obs, ci95_lo=ci95[0], ci95_hi=ci95[1], ci90_lo=ci90[0], ci90_hi=ci90[1],
                welch_t=w.statistic, welch_df=df, welch_p=w.pvalue, perm_p=pp, mw_U=mw.statistic, mw_p=mw.pvalue,
                hedges_g=hedges_g(a, b), g_lo=np.percentile(boot, 2.5), g_hi=np.percentile(boot, 97.5),
                bf01=1 / jzs_bf10(w.statistic, 9, 9), tost_p=p_tost, smallest_equiv_bound=np.max(np.abs(ci90)))


# ---------------------------------------------------------------- 1. primary item-level analysis (raw scores)
L = load("raw"); T = item_table(L).loc[ITEMS + ["cause"]]
R = [item_tests(T.loc[ITEMS, "mean"].values, "primary: raw, all four corpora")]
p = R[0]
say("=" * 100); say("SUFFICIENCY-ENTAILING (9) vs NOT SUFFICIENCY-ENTAILING (9): Hedonometer score of the effect, raw scores"); say("=" * 100)
say(f"Lines: {len(L[L.item.isin(ITEMS)]):,} ({', '.join(f'{c} {n:,}' for c, n in L[L.item.isin(ITEMS)].corpus.value_counts().items())})")
say("\nExpression means (each corpus 25%):"); say(T[["mean", "p_neg", "p_clear_neg", "p_pos", "sd", "q10", "q90"]].round(3).to_string())
say(f"\n1. ITEM-LEVEL (n = 9 vs 9 expressions)\n   group means: sufficiency {p['suf']:.3f}, not {p['not_']:.3f}; difference {p['diff']:+.3f} "
    f"[95% CI {p['ci95_lo']:+.3f}, {p['ci95_hi']:+.3f}]")
say(f"   Welch t({p['welch_df']:.1f}) = {p['welch_t']:.2f}, p = {p['welch_p']:.3f}; exact permutation p = {p['perm_p']:.3f} (48,620 relabellings); "
    f"Mann-Whitney U = {p['mw_U']:.0f}, exact p = {p['mw_p']:.3f}")
say(f"   Hedges' g = {p['hedges_g']:.2f} [bootstrap 95% CI {p['g_lo']:.2f}, {p['g_hi']:.2f}]; JZS BF01 = {p['bf01']:.2f} (evidence for no difference)")
say(f"   TOST, bounds +/-{DELTA}: p = {p['tost_p']:.4f}; 90% CI [{p['ci90_lo']:+.3f}, {p['ci90_hi']:+.3f}] -> equivalent for any bound >= {p['smallest_equiv_bound']:.3f}")
sd_pool = np.sqrt((T.loc[SUF, "mean"].var(ddof=1) + T.loc[NOT, "mean"].var(ddof=1)) / 2)
mde = TTestIndPower().solve_power(nobs1=9, ratio=1, alpha=0.05, power=0.8) * sd_pool
say(f"5. SENSITIVITY: between-expression SD within a group = {sd_pool:.3f}; smallest difference detectable with 80% power = {mde:.3f} points")
say(f"   For scale: cause = {T.loc['cause', 'mean']:.3f}, i.e. {T.loc['cause', 'mean'] - T.loc[ITEMS, 'mean'].mean():+.2f} from the 18-expression mean "
    f"({(T.loc['cause', 'mean'] - T.loc[ITEMS, 'mean'].mean()) / sd_pool:.1f} between-expression SDs)")

# ---------------------------------------------------------------- 2. mixed models
CELLS = L[L.item.isin(ITEMS)].groupby(["item", "corpus"]).agg_score.mean().rename("y").reset_index()
CELLS["group"] = pd.Categorical(np.where(CELLS.item.isin(SUF), "suf", "not"), ["not", "suf"])
with contextlib.redirect_stderr(io.StringIO()):
    m0 = smf.mixedlm("y ~ group + C(corpus)", CELLS, groups="item").fit(reml=False)
    m1 = smf.mixedlm("y ~ group * C(corpus)", CELLS, groups="item").fit(reml=False)
    mr = smf.mixedlm("y ~ group + C(corpus)", CELLS, groups="item").fit(reml=True)
b, se = mr.params["group[T.suf]"], mr.bse["group[T.suf]"]; lr = 2 * (m1.llf - m0.llf); dfi = len(m1.params) - len(m0.params)
say(f"\n2a. MIXED MODEL on the 72 expression x corpus means: y ~ group + corpus + (1 | expression), REML")
say(f"   group (suf - not) = {b:+.3f}, SE {se:.3f}, z = {b / se:.2f}, p = {mr.pvalues['group[T.suf]']:.3f}; "
    f"expression SD = {np.sqrt(mr.cov_re.iloc[0, 0]):.3f}, residual SD = {np.sqrt(mr.scale):.3f}")
say(f"   group x corpus (does the difference vary by corpus?): LR chi2({dfi}) = {lr:.2f}, p = {stats.chi2.sf(lr, dfi):.3f}")
Ll = L[L.item.isin(ITEMS)].copy(); Ll["group"] = pd.Categorical(np.where(Ll.item.isin(SUF), "suf", "not"), ["not", "suf"])
with contextlib.redirect_stderr(io.StringIO()):
    ml = smf.mixedlm("agg_score ~ group + C(corpus)", Ll, groups="item", vc_formula={"cell": "0 + C(corpus)"}).fit(reml=True)
bl, sel = ml.params["group[T.suf]"], ml.bse["group[T.suf]"]
say(f"2b. MIXED MODEL on the {len(Ll):,} lines: agg_score ~ group + corpus + (1 | expression) + (1 | expression:corpus), REML")
say(f"   group (suf - not) = {bl:+.3f}, SE {sel:.3f}, z = {bl / sel:.2f}, p = {ml.pvalues['group[T.suf]']:.3f} "
    f"(lines unweighted, so AskReddit's make/let dominate; for comparison a naive line-level t-test: p = {stats.ttest_ind(Ll.agg_score[Ll.group == 'suf'], Ll.agg_score[Ll.group == 'not']).pvalue:.1e})")

# ---------------------------------------------------------------- 3. beyond the mean
say("\n3. DISTRIBUTION MEASURES (item level, exact permutation, Holm across the six measures + two distances)")
D = []
for k in ["p_neg", "p_clear_neg", "p_pos", "sd", "q10", "q90"]:
    obs, pp = perm_p(T.loc[ITEMS, k].values); D.append(dict(measure=k, suf=T.loc[SUF, k].mean(), not_=T.loc[NOT, k].mean(), diff=obs, perm_p=pp))
M = cdf_matrix(L, ITEMS); Fs, Fn = M[:9].mean(0), M[9:].mean(0)
dist = lambda F1, F2: (np.abs(F1 - F2).sum() * 0.01, np.abs(F1 - F2).max())
w_obs, ks_obs = dist(Fs, Fn)
G1 = (SPLITS.astype(float) @ M) / 9; G2 = ((~SPLITS).astype(float) @ M) / 9
w_null, ks_null = np.abs(G1 - G2).sum(1) * 0.01, np.abs(G1 - G2).max(1)
D += [dict(measure="Wasserstein-1", suf=np.nan, not_=np.nan, diff=w_obs, perm_p=np.mean(w_null >= w_obs - 1e-12)),
      dict(measure="Kolmogorov-Smirnov D", suf=np.nan, not_=np.nan, diff=ks_obs, perm_p=np.mean(ks_null >= ks_obs - 1e-12))]
D = pd.DataFrame(D); D["p_holm"] = multipletests(D.perm_p, method="holm")[1]
say(D.round(4).to_string(index=False))
Mc = cdf_matrix(L, ["cause"])[0]
say(f"   for scale, Wasserstein-1 from the 18-expression pool: cause {dist(Mc, M.mean(0))[0]:.3f}; sufficiency group {dist(Fs, M.mean(0))[0]:.3f}")

# ---------------------------------------------------------------- 4. robustness
for sc, name in [("band", "neutral band (4-6) removed"), ("war", "Warriner valence norms")]:
    Ts = item_table(load(sc, ITEMS)).loc[ITEMS]; R.append(item_tests(Ts["mean"].values, f"scoring: {name}"))
for c in CORPORA:
    v = L[(L.corpus == c) & L.item.isin(ITEMS)].groupby("item").agg_score.mean().loc[ITEMS].values; R.append(item_tests(v, f"corpus: {c} only"))
v = T.loc[ITEMS, "mean"]
vs = v.drop("make"); a, bb = vs[[i for i in SUF if i != "make"]].values, vs[NOT].values
R.append(dict(analysis="without make (8 vs 9)", suf=a.mean(), not_=bb.mean(), diff=a.mean() - bb.mean(),
              welch_p=stats.ttest_ind(a, bb, equal_var=False).pvalue, mw_p=stats.mannwhitneyu(a, bb, method="exact").pvalue,
              ci95_lo=(a.mean() - bb.mean()) - stats.t.ppf(.975, 15) * np.sqrt(a.var(ddof=1) / 8 + bb.var(ddof=1) / 9),
              ci95_hi=(a.mean() - bb.mean()) + stats.t.ppf(.975, 15) * np.sqrt(a.var(ddof=1) / 8 + bb.var(ddof=1) / 9)))
loo = {i: (v[[j for j in SUF if j != i]].mean() if i in SUF else v[SUF].mean()) - (v[[j for j in NOT if j != i]].mean() if i in NOT else v[NOT].mean()) for i in ITEMS}
R = pd.DataFrame(R)
say("\n4. ROBUSTNESS (difference = sufficiency - not; item level)")
say(R[["analysis", "suf", "not_", "diff", "ci95_lo", "ci95_hi", "welch_p", "perm_p", "mw_p", "hedges_g", "bf01", "tost_p"]].round(3).to_string(index=False))
say(f"   leave one expression out: difference ranges {min(loo.values()):+.3f} (without {min(loo, key=loo.get)}) to {max(loo.values()):+.3f} (without {max(loo, key=loo.get)})")
R.to_csv(str(OUT) + ".tsv", sep="\t", index=False, float_format="%.5g")
D.to_csv(str(OUT) + "_distribution.tsv", sep="\t", index=False, float_format="%.5g")
T.to_csv(str(OUT) + "_items.tsv", sep="\t", float_format="%.5g")

# ---------------------------------------------------------------- figure
fig, axs = plt.subplots(1, 3, figsize=(17, 7), gridspec_kw=dict(width_ratios=[1, 1.05, 1]))
ax = axs[0]   # (a) one row per expression, sorted within group; small dots = corpus means, ring = mean of the four
yl, ylab, y = [], [], 0.0
for gk, its, gname in [("suf", SUF, "SUFFICIENCY-ENTAILING"), ("not", NOT, "NOT SUFFICIENCY-ENTAILING")]:
    vals = v[its].sort_values(ascending=False)
    yl.append(y); ylab.append(gname); y += 1
    for it, val in vals.items():
        for c in CORPORA: ax.plot(L[(L.item == it) & (L.corpus == c)].agg_score.mean(), y, "o", ms=4, color=CC[c], alpha=0.75, zorder=2)
        ax.plot(val, y, "o", ms=8, mfc="white", mec=INK, mew=1.3, zorder=3); yl.append(y); ylab.append(it); y += 1
    ci = stats.t.ppf(0.975, 8) * vals.std(ddof=1) / 3
    ax.errorbar(vals.mean(), y + 0.15, xerr=ci, fmt="D", ms=8, color=GC[gk], elinewidth=2.5, capsize=0, zorder=4)
    yl.append(y + 0.15); ylab.append(f"group mean {vals.mean():.2f}")
    y += 1.2
ax.set_yticks(yl); ax.set_yticklabels(ylab, fontsize=10)
for t in ax.get_yticklabels():
    txt = t.get_text(); t.set_fontstyle("normal" if txt.startswith("group") or txt.isupper() else "italic")
    if txt.isupper(): t.set_fontsize(8.5); t.set_color(MUTED)
ax.set_ylim(y - 0.5, -0.8); ax.set_xlim(5.15, 6.25); ax.grid(axis="x", color=GRIDC, lw=0.5, zorder=0)
ax.set_xlabel("Hedonometer score of the effect\n(expression mean, each corpus 25%)", fontsize=10, color=INK)
ax.set_title("a. The 18 expressions", loc="left", fontsize=12, color=INK, pad=24)
for c in CORPORA: ax.plot([], [], "o", ms=4, color=CC[c], label={"ententen21": "enTenTen21", "bnc": "BNC", "reddit": "AskReddit", "coca": "COCA"}[c])
ax.plot([], [], "o", ms=8, mfc="white", mec=INK, label="mean of the four")
ax.legend(fontsize=8.5, frameon=False, loc="lower left", ncol=5, columnspacing=0.8, handletextpad=0.2, bbox_to_anchor=(-0.02, 0.995))

ax = axs[1]   # (b) forest plot of the difference
F = R.copy(); F["lab"] = F.analysis.str.replace("primary: raw, all four corpora", "Primary (raw scores, four corpora)").str.replace("scoring: ", "").str.replace("corpus: ", "").str.replace("ententen21", "enTenTen21").str.replace("bnc", "BNC").str.replace("reddit", "AskReddit").str.replace("coca", "COCA")
yy = np.arange(len(F))[::-1]
ax.axvspan(-DELTA, DELTA, color=GRIDC, zorder=0); ax.axvline(0, color=MUTED, lw=0.8)
for y, (_, r) in zip(yy, F.iterrows()):
    prim = r.analysis.startswith("primary")
    ax.plot([r.ci95_lo, r.ci95_hi], [y, y], color=INK, lw=2.4 if prim else 1.4, solid_capstyle="round")
    ax.plot(r["diff"], y, "D" if prim else "o", ms=8 if prim else 6, color=INK if prim else "white", mec=INK, mew=1.2, zorder=3)
ax.set_yticks(yy); ax.set_yticklabels(F.lab, fontsize=9.5); ax.set_xlim(-0.45, 0.45)
ax.set_xlabel("Difference: sufficiency − not sufficiency (95% CI)", fontsize=10, color=INK)
ax.text(0, yy.max() + 0.8, f"equivalence bounds ±{DELTA}", ha="center", fontsize=8.5, color=MUTED)
ax.set_ylim(-0.7, yy.max() + 1.4); ax.set_title("b. Group difference, primary and robustness", loc="left", fontsize=12, color=INK)

ax = axs[2]   # (c) whole distributions
ax.plot(GRID, Fs, color=GC["suf"], lw=2, label="Clearly sufficiency-entailing")
ax.plot(GRID, Fn, color=GC["not"], lw=2, ls=(0, (5, 2)), label="Clearly not sufficiency-entailing")
ax.plot(GRID, Mc, color=MUTED, lw=1.3, ls=":", label="cause (for scale)")
ax.axvline(5, color=MUTED, lw=0.6, ls=":")
ax.set_xlim(1, 9); ax.set_ylim(0, 1); ax.set_xlabel("Hedonometer score of the effect (per line)", fontsize=10, color=INK)
ax.set_ylabel("Cumulative share of lines", fontsize=10, color=INK)
ax.legend(fontsize=9, frameon=False, loc="upper left")
ax.text(8.9, 0.04, f"Wasserstein-1 = {w_obs:.3f}, permutation p = {D.set_index('measure').loc['Wasserstein-1', 'perm_p']:.2f}\n"
        f"KS D = {ks_obs:.3f}, permutation p = {D.set_index('measure').loc['Kolmogorov-Smirnov D', 'perm_p']:.2f}", ha="right", va="bottom", fontsize=8.5, color=INK)
ax.set_title("c. Whole distributions", loc="left", fontsize=12, color=INK)
for a_ in axs:
    for s in ["top", "right"]: a_.spines[s].set_visible(False)
    a_.tick_params(colors=MUTED, labelsize=9)
axs[2].grid(color=GRIDC, lw=0.5)
fig.subplots_adjust(left=0.135, right=0.99, top=0.88, bottom=0.12, wspace=0.62)
fig.savefig(str(OUT) + ".png", dpi=200); fig.savefig(str(OUT) + ".pdf")
open(str(OUT) + ".txt", "w").write(log.getvalue()); print(log.getvalue())

# ---------------------------------------------------------------- 6. cause compared with the two groups (10 Nov 2026, Dean)
log = io.StringIO()
def crawford_howell(x, s):
    """Single case x vs a small control sample s (Crawford & Howell 1998); returns t, df, two-sided p, and the
    estimated share of the population of such expressions with a lower score (Crawford & Garthwaite 2002)."""
    n = len(s); t = (x - s.mean()) / (s.std(ddof=1) * np.sqrt((n + 1) / n)); df = n - 1
    return t, df, 2 * stats.t.sf(abs(t), df), stats.t.cdf(t, df)
say("=" * 100); say("6. CAUSE COMPARED WITH THE TWO GROUPS (raw scores, each corpus 25%)"); say("=" * 100)
xc = T.loc["cause", "mean"]
for name, its in [("sufficiency-entailing (9)", SUF), ("not sufficiency-entailing (9)", NOT), ("all 18", ITEMS)]:
    s = T.loc[its, "mean"].values; t, df, pp, share = crawford_howell(xc, s)
    say(f"   cause {xc:.3f} vs {name}: mean {s.mean():.3f}, difference {xc - s.mean():+.3f} = {(xc - s.mean()) / s.std(ddof=1):.1f} SDs of that group; "
        f"lowest member {s.min():.3f}; Crawford-Howell t({df}) = {t:.2f}, p = {pp:.1e}; estimated share of such expressions below cause = {share:.1e}")
say("\n   Per corpus (corpus means; rank of cause among the 19):")
for c in CORPORA:
    m = L[L.corpus == c].groupby("item").agg_score.mean()
    say(f"   {c:11s} cause {m['cause']:.3f}; sufficiency {m[SUF].mean():.3f}; not {m[NOT].mean():.3f}; "
        f"gap to the lowest other expression ({m[ITEMS].idxmin()}) {m['cause'] - m[ITEMS].min():+.3f}; rank {int(m[ITEMS + ['cause']].rank().loc['cause'])}/19")
say("\n   Distribution measures (item level; groups = mean of their 9 expressions):")
K = ["mean", "p_neg", "p_clear_neg", "p_pos", "sd", "q10", "q90"]
Dc = pd.DataFrame({"cause": T.loc["cause", K], "sufficiency": T.loc[SUF, K].mean(), "not": T.loc[NOT, K].mean(),
                   "range of the 18": [f"{T.loc[ITEMS, k].min():.3f}-{T.loc[ITEMS, k].max():.3f}" for k in K]})
Lc = L[L.item == "cause"]; med = lambda g: np.median(g.agg_score)
say(Dc.round(3).to_string())
say(f"   share of lines scoring 2-3.5 (the lower mode of cause): cause {Lc.w[(Lc.agg_score >= 2) & (Lc.agg_score < 3.5)].sum():.3f} "
    f"vs groups {np.mean([L[L.item == i].w[(L[L.item == i].agg_score >= 2) & (L[L.item == i].agg_score < 3.5)].sum() for i in SUF]):.3f} / "
    f"{np.mean([L[L.item == i].w[(L[L.item == i].agg_score >= 2) & (L[L.item == i].agg_score < 3.5)].sum() for i in NOT]):.3f}")
for name, F_ in [("sufficiency", Fs), ("not", Fn)]:
    d = dist(Mc, F_); say(f"   cause vs {name} group distribution: Wasserstein-1 = {d[0]:.3f}, KS D = {d[1]:.3f}  (sufficiency vs not: {w_obs:.3f}, {ks_obs:.3f})")
for sc in ["band", "war"]:
    Ts = item_table(load(sc)).loc[ITEMS + ["cause"]]; s = Ts.loc[ITEMS, "mean"].values; t, df, pp, *_ = crawford_howell(Ts.loc["cause", "mean"], s)
    say(f"   scoring {sc}: cause {Ts.loc['cause','mean']:.3f} vs all 18 {s.mean():.3f} (lowest {s.min():.3f}); Crawford-Howell t({df}) = {t:.2f}, p = {pp:.1e}")
Pc = pd.read_csv(config.RESULTS / "pvalues_all.tsv", sep="\t")
Pc = Pc[(Pc.scoring == "raw") & (Pc.measure == "NV") & (Pc.polarity == "combined") & (Pc.baseline_type == "verbs") & (Pc.item == "cause")]
say("\n   cause vs each corpus's baseline (all other verbs; from pvalues_all.tsv):")
say(Pc[["corpus", "n", "mean", "baseline", "diff", "p_holm"]].to_string(index=False))
open(str(OUT) + "_cause.txt", "w").write(log.getvalue()); print(log.getvalue())

# ---------------------------------------------------------------- 7. cause vs because (10 Nov 2026, Dean)
log = io.StringIO()
say("=" * 100); say("7. CAUSE vs BECAUSE (raw scores; because = 'because (clause)'; each corpus 25%)"); say("=" * 100)
Lr = pd.read_pickle(config.RESULTS / "lines_raw.pkl")
CB = Lr.loc[Lr.agg_score.notna() & Lr.item.isin(["cause", "because (clause)"]), ["corpus", "item", "agg_score", "N_keys", "V_keys"]].copy()
CB["key"] = CB.N_keys.fillna("") + "|" + CB.V_keys.fillna(""); CB["is_cause"] = CB.item == "cause"
say("   lines: " + "; ".join(f"{c}: cause {((CB.corpus == c) & CB.is_cause).sum():,}, because {((CB.corpus == c) & ~CB.is_cause).sum():,}" for c in CORPORA))
def wmean(df): return np.mean([df[df.corpus == c].agg_score.mean() for c in CORPORA])
mc, mb = wmean(CB[CB.is_cause]), wmean(CB[~CB.is_cause]); diff = mc - mb
per = pd.DataFrame({c: [CB[(CB.corpus == c) & CB.is_cause].agg_score.mean(), CB[(CB.corpus == c) & ~CB.is_cause].agg_score.mean()] for c in CORPORA}, index=["cause", "because"]).T
per["diff"] = per.cause - per.because
say(f"\n   means: cause {mc:.3f}, because {mb:.3f}; difference {diff:+.3f}")
say("   per corpus:\n" + per.round(3).to_string())
tt = stats.ttest_1samp(per["diff"], 0)
say(f"   corpus-level: difference negative in {int((per['diff'] < 0).sum())}/4 corpora; range {per['diff'].min():+.3f} to {per['diff'].max():+.3f}; "
    f"paired t(3) = {tt.statistic:.2f}, p = {tt.pvalue:.4f}")
# cluster bootstrap: resample effect words (N|V keys) with replacement, separately within each corpus, shared by both expressions
B = 2000; boots = []
grp = {c: CB[CB.corpus == c].groupby("key") for c in CORPORA}
keyl = {c: np.array(list(grp[c].groups.keys())) for c in CORPORA}
agg = {c: CB[CB.corpus == c].groupby(["key", "is_cause"]).agg_score.agg(["sum", "size"]).unstack(fill_value=0) for c in CORPORA}
for _ in range(B):
    d = []
    for c in CORPORA:
        A = agg[c]; idx = rng.integers(0, len(A), len(A)); S = A.iloc[idx].sum()
        d.append(S[("sum", True)] / S[("size", True)] - S[("sum", False)] / S[("size", False)])
    boots.append(np.mean(d))
lo, hi = np.percentile(boots, [2.5, 97.5])
say(f"   95% CI for the difference, bootstrap over effect words (2,000 resamples, within corpus): [{lo:+.3f}, {hi:+.3f}]")
# effect sizes on the lines (corpus-weighted)
CB["w"] = 0.25 / CB.groupby(["item", "corpus"]).agg_score.transform("size")
def wsd(df): m = np.average(df.agg_score, weights=df.w); return np.sqrt(np.average((df.agg_score - m) ** 2, weights=df.w))
sdc, sdb = wsd(CB[CB.is_cause]), wsd(CB[~CB.is_cause]); d_line = diff / np.sqrt((sdc ** 2 + sdb ** 2) / 2)
ps = []
for c in CORPORA:
    a = CB[(CB.corpus == c) & CB.is_cause].agg_score.values; b = CB[(CB.corpus == c) & ~CB.is_cause].agg_score.values
    ps.append(stats.mannwhitneyu(b, a).statistic / (len(a) * len(b)))
say(f"   line-level Cohen's d = {d_line:.2f} (SDs: cause {sdc:.2f}, because {sdb:.2f})")
say(f"   probability of superiority (a random because-effect scores higher than a random cause-effect; ties count half): "
    f"{np.mean(ps):.3f} (per corpus {', '.join(f'{p:.3f}' for p in ps)})")
Tcb = item_table(load("raw", ["cause", "because (clause)"])).loc[["cause", "because (clause)"]]
say("\n   distribution measures:\n" + Tcb[["mean", "p_neg", "p_clear_neg", "p_pos", "sd", "q10", "q90"]].round(3).to_string())
lower = lambda df, lo_, hi_: df.w[(df.agg_score >= lo_) & (df.agg_score < hi_)].sum()
Lw = load("raw", ["cause", "because (clause)"])
say(f"   share of lines in 2-3.5: cause {lower(Lw[Lw.item == 'cause'], 2, 3.5):.3f}, because {lower(Lw[Lw.item == 'because (clause)'], 2, 3.5):.3f}")
say(f"   relative to the 18 expressions (mean {T.loc[ITEMS, 'mean'].mean():.3f}, between-expression SD {T.loc[ITEMS, 'mean'].std(ddof=1):.3f}): "
    f"cause {mc - T.loc[ITEMS, 'mean'].mean():+.3f} ({(mc - T.loc[ITEMS, 'mean'].mean()) / T.loc[ITEMS, 'mean'].std(ddof=1):+.1f} SD), "
    f"because {mb - T.loc[ITEMS, 'mean'].mean():+.3f} ({(mb - T.loc[ITEMS, 'mean'].mean()) / T.loc[ITEMS, 'mean'].std(ddof=1):+.1f} SD)")
s18 = T.loc[ITEMS, "mean"].values; t, df_, pp, share = crawford_howell(mb, s18)
say(f"   because vs the 18 (Crawford-Howell): t({df_}) = {t:.2f}, p = {pp:.3f}")
for sc in ["band", "war"]:
    Ts = item_table(load(sc, ["cause", "because (clause)"])); say(f"   scoring {sc}: cause {Ts.loc['cause','mean']:.3f}, because {Ts.loc['because (clause)','mean']:.3f}, difference {Ts.loc['cause','mean'] - Ts.loc['because (clause)','mean']:+.3f}")
open(str(OUT) + "_cause_because.txt", "w").write(log.getvalue()); print(log.getvalue())

# ---------------------------------------------------------------- 8. pairwise comparisons of single expressions (10 Nov 2026, Dean)
from statsmodels.stats.anova import AnovaRM
def compare(items, title, fname, B=2000):
    """Pairwise comparison of single expressions: corpus-weighted means, per-corpus differences (paired t over the
    4 corpora), 95% CI by bootstrapping effect words (N|V keys) within each corpus (keys shared by the expressions are
    resampled together), line-level Cohen's d, probability of superiority, distribution measures. With > 2 expressions:
    repeated-measures ANOVA and Friedman test over corpora, Holm across the pairs."""
    global log; log = io.StringIO()
    X = Lr.loc[Lr.agg_score.notna() & Lr.item.isin(items), ["corpus", "item", "agg_score", "N_keys", "V_keys"]].copy()
    X["key"] = X.N_keys.fillna("") + "|" + X.V_keys.fillna("")
    say("=" * 100); say(title + " (raw scores, each corpus 25%)"); say("=" * 100)
    per = X.groupby(["item", "corpus"]).agg_score.mean().unstack()[CORPORA].loc[items]; n = X.groupby(["item", "corpus"]).size().unstack()[CORPORA].loc[items]
    per["four"] = per[CORPORA].mean(1); say("   means (n lines in brackets):")
    say(pd.DataFrame({c: [f"{per.loc[i, c]:.3f} ({n.loc[i, c]:,})" for i in items] for c in CORPORA}, index=items).assign(four=per.four.round(3)).to_string())
    S = {}; N = {}
    for c in CORPORA:
        A = X[X.corpus == c].groupby(["key", "item"]).agg_score.agg(["sum", "size"]).unstack(fill_value=0)
        S[c] = A["sum"].reindex(columns=items, fill_value=0).values; N[c] = A["size"].reindex(columns=items, fill_value=0).values
    boot = np.zeros((B, len(items)))
    for b in range(B):
        mm = []
        for c in CORPORA:
            idx = rng.integers(0, len(S[c]), len(S[c])); mm.append(S[c][idx].sum(0) / N[c][idx].sum(0))
        boot[b] = np.mean(mm, axis=0)
    X["w"] = 0.25 / X.groupby(["item", "corpus"]).agg_score.transform("size")
    Ti = item_table(X.assign()).loc[items]
    if len(items) > 2:
        long = per[CORPORA].stack().rename("y").reset_index()
        a = AnovaRM(long, "y", "corpus", within=["item"]).fit().anova_table.iloc[0]
        fr = stats.friedmanchisquare(*[per.loc[i, CORPORA].values for i in items])
        say(f"\n   omnibus over the 4 corpora: repeated-measures ANOVA F({a['Num DF']:.0f}, {a['Den DF']:.0f}) = {a['F Value']:.1f}, p = {a['Pr > F']:.1e}; "
            f"Friedman chi2({len(items) - 1}) = {fr.statistic:.2f}, p = {fr.pvalue:.3f} (with 4 corpora Friedman cannot go below p ~ .002)")
        say("   order by mean in each corpus: " + "; ".join(f"{c}: {' < '.join(per[c].sort_values().index)}" for c in CORPORA))
    rows = []
    for i, j in itertools.combinations(range(len(items)), 2):
        a_, b_ = items[i], items[j]; d = per.loc[a_, CORPORA] - per.loc[b_, CORPORA]; tt = stats.ttest_1samp(d.values, 0)
        lo, hi = np.percentile(boot[:, i] - boot[:, j], [2.5, 97.5])
        xa, xb = X[X.item == a_], X[X.item == b_]
        def wsd(df): m = np.average(df.agg_score, weights=df.w); return np.sqrt(np.average((df.agg_score - m) ** 2, weights=df.w))
        dl = (per.loc[a_, "four"] - per.loc[b_, "four"]) / np.sqrt((wsd(xa) ** 2 + wsd(xb) ** 2) / 2)
        ps = np.mean([stats.mannwhitneyu(xb[xb.corpus == c].agg_score, xa[xa.corpus == c].agg_score).statistic / ((xa.corpus == c).sum() * (xb.corpus == c).sum()) for c in CORPORA])
        rows.append(dict(pair=f"{a_} - {b_}", diff=per.loc[a_, "four"] - per.loc[b_, "four"], ci_lo=lo, ci_hi=hi, boot_excludes_0=(lo > 0) or (hi < 0),
                         same_sign=f"{int((np.sign(d) == np.sign(d.mean())).sum())}/4", corpus_min=d.min(), corpus_max=d.max(), paired_t3=tt.statistic, p=tt.pvalue,
                         cohens_d=dl, P_second_higher=ps))
    R_ = pd.DataFrame(rows)
    if len(R_) > 1: R_["p_holm"] = multipletests(R_.p, method="holm")[1]
    say("\n   pairwise (diff = first - second; CI = bootstrap over effect words; p = paired t over the 4 corpora; "
        "P_second_higher = chance a random effect of the second expression scores higher than one of the first):")
    say(R_.round(3).to_string(index=False))
    say("\n   distribution measures:\n" + Ti[["mean", "p_neg", "p_clear_neg", "p_pos", "sd", "q10", "q90"]].round(3).to_string())
    for sc in ["band", "war"]:
        Ts = item_table(load(sc, items)).loc[items, "mean"]; say(f"   scoring {sc}: " + ", ".join(f"{i} {Ts[i]:.3f}" for i in items))
    open(str(OUT) + f"_{fname}.txt", "w").write(log.getvalue()); print(log.getvalue()); return R_

compare(["sufficient for", "necessary for"], "8a. SUFFICIENT FOR vs NECESSARY FOR", "sufficient_necessary")
compare(["cause", "make", "force", "let"], "8b. CAUSE vs MAKE vs FORCE vs LET (make = causative frames only)", "cause_make_force_let")

compare(["cause", "the cause of", "result from", "arise from", "stem from"], "8c. CAUSE vs ITS NEAREST NEIGHBOURS AMONG THE 101 EXPRESSIONS", "cause_neighbours")
