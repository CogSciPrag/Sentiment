"""[cause and make only; 3 Nov 2026, Dean] Word-shift graphs (Dodds et al. 2011; Reagan et al. 2017): which effect words make one set of effects happier or
sadder than a reference set. Uses the scored effect words (N and V words with a Hedonometer score, raw).
Contribution of word i: (h_i - h_ref) * (p_comp,i - p_ref,i), as % of |h_comp - h_ref|.
Averages here are over word tokens (as in the word-shift method), so they differ slightly from the line means.
Input: data/results/extra/ (step 10). Output: data/figures/. Output: fig_wordshift_cause_make_askreddit / _all_corpora (.png/.pdf) and extra/wordshift_<item>_vs_other_verbs_<corpus>.tsv.
Usage: python figures/fig_wordshift_cause_make.py
"""
import numpy as np, pandas as pd, matplotlib
from causesent import config
import causesent.style  # Helvetica
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collections import Counter
from causesent.scoring import H
EXTRA = str(config.RESULTS / "extra") + "/"; FIG = str(config.FIGURES) + "/"; config.ensure_dirs()

A = pd.read_pickle(EXTRA + "all_lines.pkl")

def freqs(D):
    c = Counter()
    for col in (["N_keys", "V_keys"] if "N_keys" in D else ["N_words", "V_words"]):   # looked-up words (form-first)
        for s in D[col].dropna():
            c.update(w for w in str(s).split() if w in H)
    tot = sum(c.values())
    return {w: n / tot for w, n in c.items()}, tot

def shift(comp, ref):
    pc, nc = freqs(comp); pr, nr = freqs(ref)
    hc = sum(p * H[w] for w, p in pc.items()); hr = sum(p * H[w] for w, p in pr.items())
    words = set(pc) | set(pr)
    d = {w: (H[w] - hr) * (pc.get(w, 0) - pr.get(w, 0)) for w in words}
    tot = abs(hc - hr)
    df = pd.DataFrame({"word": list(d), "contrib_pct": [100 * v / tot for v in d.values()],
                       "h": [H[w] for w in d], "p_comp": [pc.get(w, 0) for w in d], "p_ref": [pr.get(w, 0) for w in d]})
    df["abs"] = df.contrib_pct.abs()
    return df.sort_values("abs", ascending=False), hc, hr, nc, nr


BASEI = "BASELINE (other verbs)"
NAMES = {"ententen21": "enTenTen21", "bnc": "BNC", "coca": "COCA", "reddit": "AskReddit"}
TOP = 30
INK = "#52514e"

def panel(ax, comp, ref, title, small=False, trim_right=False, left=None):
    df, hc, hr, nc, nr = shift(comp, ref)
    t = df.head(TOP).iloc[::-1]
    up = t.p_comp > t.p_ref; pos = t.h > hr
    colors = ["#1b7837" if p and u else "#a6d49f" if p else "#b2182b" if u else "#f1a99b" for p, u in zip(pos, up)]
    y = np.arange(len(t))
    ax.barh(y, t.contrib_pct, color=colors, edgecolor="#0b0b0b", linewidth=0.3)
    for yi, (_, r), p, u in zip(y, t.iterrows(), pos, up):
        sym = ("+" if p else "\u2212") + ("\u2191" if u else "\u2193")
        x = r.contrib_pct
        ax.text(x + (0.3 if x >= 0 else -0.3), yi, f"{r.word} {sym}", va="center", ha="left" if x >= 0 else "right", fontsize=6.8 if small else 8)
    ax.axvline(0, color=INK, lw=0.8)
    lim = max(abs(t.contrib_pct).max() * 1.9, 5); ax.set_xlim(-lim, lim)
    if trim_right and t.contrib_pct.max() <= 0: ax.set_xlim(-(left if left else lim * 0.97), lim * 0.02)   # no bar to the right of 0: no empty space there
    ax.set_yticks([]); ax.set_ylim(-0.7, len(t) - 0.3)
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=7 if small else 8)
    ax.set_xlabel("contribution to the difference (% of |difference|)", fontsize=7.5 if small else 8.5, color=INK)
    ax.set_title(title, fontsize=8.6 if small else 9.5, loc="left")
    return df

def get(c, it):
    return A[(A.corpus == c) & (A.item == it)]

FOOT = ("Bars right of 0 make the effects of the verb happier than those of all other verbs, bars left make them sadder. "
        "+\u2191 happier-than-reference word used more; \u2212\u2193 sadder word used less (both raise the score); "
        "\u2212\u2191 sadder word used more; +\u2193 happier word used less (both lower it).\n"
        "Dark bars = word used more often with the verb; light bars = used less often. Top 30 words by absolute contribution. "
        "Averages are over effect words (tokens), so they differ slightly from the line means. make: causative frames only. "
        "Where a verb's mean is close to the baseline (make, force) the difference is small, so single words can account for large percentages (and for more than 100% in either direction). "
        "Method: Dodds et al. (2011), Reagan et al. (2017).")

import textwrap
def run(corpora, fname, figsize, small, suptitle, wrap):
    fig, axes = plt.subplots(2, len(corpora), figsize=figsize, squeeze=False)
    for j, c in enumerate(corpora):
        for i, it in enumerate(["cause", "make"]):
            comp, ref = get(c, it), get(c, BASEI)
            df, hc, hr, nc, nr = shift(comp, ref)
            lab = "ABCDEFGH"[i * len(corpora) + j]
            title = (f"{lab}. {it} vs all other verbs ({NAMES[c]})\n{it}: {hc:.2f} ({nc:,} effect words)   other verbs: {hr:.2f} ({nr:,})")
            d = panel(axes[i, j], comp, ref, title, small)
            d.assign(comparison=title).to_csv(f"{EXTRA}wordshift_{it}_vs_other_verbs_{c}.tsv", sep="\t", index=False, float_format="%.5f")
    fig.suptitle(suptitle, fontsize=13, x=0.01, ha="left")
    fig.text(0.01, 0.005, textwrap.fill(FOOT.replace(chr(10), ' '), wrap), fontsize=7.4 if small else 7.6, color=INK, va="bottom")
    fig.tight_layout(rect=(0, 0.07 if small else 0.08, 1, 0.96))
    fig.savefig(FIG + fname + ".png", dpi=200); fig.savefig(FIG + fname + ".pdf"); plt.close(fig); print("saved", fname)

run(["reddit"], "fig_wordshift_cause_make_askreddit", (11, 11), False,
    "Word-shift graphs: which effect words make cause and make differ from other verbs (AskReddit, raw Hedonometer scores)", 150)
run(["ententen21", "bnc", "coca", "reddit"], "fig_wordshift_cause_make_all_corpora", (22, 14), True,
    "Word-shift graphs: which effect words make cause and make differ from other verbs, per corpus (raw Hedonometer scores)", 330)

def run_side(c, fname, suptitle, wrap):
    """One corpus, cause (left) and make (right) side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 8.6))
    for j, it in enumerate(["cause", "make"]):
        comp, ref = get(c, it), get(c, BASEI)
        df, hc, hr, nc, nr = shift(comp, ref)
        title = (f"{'AB'[j]}. {it} vs all other verbs ({NAMES[c]})\n{it}: {hc:.2f} ({nc:,} effect words)   other verbs: {hr:.2f} ({nr:,})")
        panel(axes[j], comp, ref, title, False, trim_right=(j == 0), left=(10 if j == 0 else None))
        if j == 1: axes[j].set_xlim(-100, 100)   # B: axis cut at -100 and 100 (Dean, 3 Nov)
    fig.suptitle(suptitle, fontsize=13, x=0.01, ha="left")
    fig.text(0.01, 0.005, textwrap.fill(FOOT.replace(chr(10), " "), wrap), fontsize=7.8, color=INK, va="bottom")
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(FIG + fname + ".png", dpi=200); fig.savefig(FIG + fname + ".pdf"); plt.close(fig); print("saved", fname)

run_side("reddit", "fig_wordshift_cause_make_askreddit_side_by_side",
         "Word-shift graphs: which effect words make cause and make differ from other verbs (AskReddit, raw Hedonometer scores)", 165)


def run_side3(c, fname, items, suptitle, wrap, figsize=(18.5, 8.6)):
    """One corpus, several verbs side by side (A, B, C ...). Axis rules: panel A cut at -10 and no space right of 0 (all bars negative); B at +-100 (Dean, 3 Nov); others: symmetric, then trimmed when all bars point one way."""
    fig, axes = plt.subplots(1, len(items), figsize=figsize)
    for j, it in enumerate(items):
        comp, ref = get(c, it), get(c, BASEI)
        df, hc, hr, nc, nr = shift(comp, ref)
        title = (f"{'ABCDEF'[j]}. {it} vs all other verbs ({NAMES[c]})\n{it}: {hc:.2f} ({nc:,} effect words)   other verbs: {hr:.2f} ({nr:,})")
        panel(axes[j], comp, ref, title, False, trim_right=(it == "cause"), left=(10 if it == "cause" else None))
        if it == "make": axes[j].set_xlim(-100, 100)
        d = df.head(TOP); print(it, "n effect words", nc, "mean", round(hc, 3), "| top-30 contributions range", round(d.contrib_pct.min(), 1), round(d.contrib_pct.max(), 1))
    fig.suptitle(suptitle, fontsize=13, x=0.01, ha="left")
    fig.text(0.01, 0.005, textwrap.fill(FOOT.replace(chr(10), " "), wrap), fontsize=7.8, color=INK, va="bottom")
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(FIG + fname + ".png", dpi=200); fig.savefig(FIG + fname + ".pdf"); plt.close(fig); print("saved", fname)

run_side3("reddit", "fig_wordshift_cause_make_force_askreddit_side_by_side", ["cause", "make", "force"],
          "Word-shift graphs: which effect words make cause, make and force differ from other verbs (AskReddit, raw Hedonometer scores)", 230)


# ---------------------------------------------------------------------------------------------------------------
# All four corpora, equal weight (4 Nov 2026, Dean): the word frequencies p of a set (the verb, or all other verbs) are the
# AVERAGE of the four corpora's own frequencies, so every corpus contributes 25% whatever its size (AskReddit has about 20 times
# more effect words than COCA).  Same word-shift formula as above, on these averaged distributions.
CORPORA = ["ententen21", "bnc", "coca", "reddit"]

def freqs_equal(item):
    ps, ns = [], []
    for c in CORPORA:
        p, n = freqs(get(c, item)); ps.append(p); ns.append(n)
    words = set().union(*ps)
    return {w: sum(p.get(w, 0) for p in ps) / len(CORPORA) for w in words}, ns

def shift_equal(item, ref=BASEI):
    pc, nc = freqs_equal(item); pr, nr = freqs_equal(ref)
    hc = sum(p * H[w] for w, p in pc.items()); hr = sum(p * H[w] for w, p in pr.items())
    d = {w: (H[w] - hr) * (pc.get(w, 0) - pr.get(w, 0)) for w in set(pc) | set(pr)}
    tot = abs(hc - hr)
    df = pd.DataFrame({"word": list(d), "contrib_pct": [100 * v / tot for v in d.values()], "h": [H[w] for w in d],
                       "p_comp": [pc.get(w, 0) for w in d], "p_ref": [pr.get(w, 0) for w in d]})
    df["abs"] = df.contrib_pct.abs()
    return df.sort_values("abs", ascending=False), hc, hr, nc, nr

def panel_equal(ax, item, title, left=None, right=None):
    df, hc, hr, nc, nr = shift_equal(item)
    t = df.head(TOP).iloc[::-1]
    up = t.p_comp > t.p_ref; pos = t.h > hr
    colors = ["#1b7837" if p and u else "#a6d49f" if p else "#b2182b" if u else "#f1a99b" for p, u in zip(pos, up)]
    y = np.arange(len(t))
    ax.barh(y, t.contrib_pct, color=colors, edgecolor="#0b0b0b", linewidth=0.3)
    for yi, (_, r), p, u in zip(y, t.iterrows(), pos, up):
        sym = ("+" if p else "\u2212") + ("\u2191" if u else "\u2193"); x = r.contrib_pct
        ax.text(x + (0.3 if x >= 0 else -0.3), yi, f"{r.word} {sym}", va="center", ha="left" if x >= 0 else "right", fontsize=8)
    ax.axvline(0, color=INK, lw=0.8)
    lim = max(abs(t.contrib_pct).max() * 1.9, 5); ax.set_xlim(-lim, lim)
    if left or right: ax.set_xlim(-(left if left else lim), (right if right else lim))
    ax.set_yticks([]); ax.set_ylim(-0.7, len(t) - 0.3)
    for sp in ["top", "right", "left"]: ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8)
    ax.set_xlabel("contribution to the difference (% of |difference|)", fontsize=8.5, color=INK)
    ax.set_title(title.format(hc=hc, hr=hr), fontsize=9.5, loc="left")
    df.to_csv(f"{EXTRA}wordshift_{item}_vs_other_verbs_four_corpora_equal_weight.tsv", sep="\t", index=False, float_format="%.5f")
    return df, hc, hr, nc, nr

def run_equal(fname, items, suptitle, wrap, figsize=(18.5, 8.6), axes_cfg=None):
    axes_cfg = axes_cfg or {}
    fig, axes = plt.subplots(1, len(items), figsize=figsize)
    for j, it in enumerate(items):
        title = f"{'ABCDEF'[j]}. {it} vs all other verbs (four corpora, equal weight)\n{it}: {{hc:.2f}}   other verbs: {{hr:.2f}}"
        df, hc, hr, nc, nr = panel_equal(axes[j], it, title, **axes_cfg.get(it, {}))
        print(it, "mean %.3f vs %.3f" % (hc, hr), "effect words per corpus", nc, "| top-30 range", round(df.head(TOP).contrib_pct.min(), 1), round(df.head(TOP).contrib_pct.max(), 1))
    fig.suptitle(suptitle, fontsize=13, x=0.01, ha="left")
    foot = (FOOT.replace(chr(10), " ").replace("Averages are over effect words (tokens), so they differ slightly from the line means.",
            "Each of the four corpora (enTenTen21, BNC, COCA, AskReddit) contributes 25% of the effect words of the verb and 25% of those of the other verbs "
            "(each corpus's word frequencies are averaged), so the means here are not the line means or the pooled means."))
    fig.text(0.01, 0.005, textwrap.fill(foot, wrap), fontsize=7.8, color=INK, va="bottom")
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(FIG + fname + ".png", dpi=200); fig.savefig(FIG + fname + ".pdf"); plt.close(fig); print("saved", fname)

run_equal("fig_wordshift_cause_make_force_four_corpora_equal_weight", ["cause", "make", "force"],
          "Word-shift graphs: which effect words make cause, make and force differ from other verbs (four corpora, each weighted 25%, raw Hedonometer scores)", 230,
          axes_cfg={"cause": dict(left=10, right=0.3), "make": dict(left=100, right=100)})   # same axis cuts as the AskReddit version (A: -10 to 0; B: +-100)
