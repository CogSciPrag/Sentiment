"""Word-shift graphs (Dodds et al. 2011; Reagan et al. 2017): which effect words make one set of effects happier or
sadder than a reference set. Uses the scored effect words (N and V words with a Hedonometer score, raw).
Contribution of word i: (h_i - h_ref) * (p_comp,i - p_ref,i), as % of |h_comp - h_ref|.
Averages here are over word tokens (as in the word-shift method), so they differ slightly from the line means.
Input: data/results/extra/ (step 10). Output: data/figures/. Usage: python figures/fig_wordshift.py
"""
import numpy as np, pandas as pd, matplotlib
from causesent import config
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collections import Counter
from causesent.scoring import H
EXTRA = str(config.RESULTS / "extra") + "/"; FIG = str(config.FIGURES) + "/"; config.ensure_dirs()

A = pd.read_pickle(EXTRA + "all_lines.pkl")

def freqs(D):
    c = Counter()
    for col in ["N_words", "V_words"]:
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

BASE = A[(A.item == "BASELINE (other verbs)") & (A.corpus == "reddit")]
R = A[(A.corpus == "reddit")]; ENT = A[A.corpus == "ententen21"]
comps = [("A. cause vs all other verbs (AskReddit)", R[R.item == "cause"], BASE, "cause", "other verbs"),
         ("B. ensure vs all other verbs (AskReddit)", R[R.item == "ensure"], BASE, "ensure", "other verbs"),
         ("C. guarantee vs all other verbs (AskReddit)", R[R.item == "guarantee"], BASE, "guarantee", "other verbs"),
         ("D. ensure vs cause (enTenTen21)", ENT[ENT.item == "ensure"], ENT[ENT.item == "cause"], "ensure", "cause")]
TOP = 30
fig, axes = plt.subplots(1, 4, figsize=(20, 11))
out = []
for ax, (title, comp, ref, cn, rn) in zip(axes, comps):
    df, hc, hr, nc, nr = shift(comp, ref)
    t = df.head(TOP).iloc[::-1]
    up = t.p_comp > t.p_ref; pos = t.h > hr
    colors = ["#1b7837" if p and u else "#a6d49f" if p else "#b2182b" if u else "#f1a99b" for p, u in zip(pos, up)]
    y = np.arange(len(t))
    ax.barh(y, t.contrib_pct, color=colors, edgecolor="#0b0b0b", linewidth=0.3)
    for yi, (_, r), p, u in zip(y, t.iterrows(), pos, up):
        sym = ("+" if p else "−") + ("↑" if u else "↓")
        x = r.contrib_pct
        ax.text(x + (0.3 if x >= 0 else -0.3), yi, f"{r.word} {sym}", va="center", ha="left" if x >= 0 else "right", fontsize=8)
    ax.axvline(0, color="#52514e", lw=0.8)
    lim = max(abs(t.contrib_pct).max() * 1.9, 5); ax.set_xlim(-lim, lim)
    ax.set_yticks([]); ax.set_ylim(-0.7, len(t) - 0.3)
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.set_xlabel("contribution to the difference (% of |difference|)", fontsize=8.5, color="#52514e")
    ax.set_title(f"{title}\n{cn}: {hc:.2f} ({nc:,} effect words)   {rn}: {hr:.2f} ({nr:,})", fontsize=9.5, loc="left")
    df.assign(comparison=title).to_csv(f"{EXTRA}wordshift_{cn}_vs_{rn.replace(' ', '_')}.tsv", sep="\t", index=False, float_format="%.5f")
fig.suptitle("Word-shift graphs: which effect words drive the difference in sentiment (raw Hedonometer scores)", fontsize=13, x=0.01, ha="left")
fig.text(0.01, 0.005, "Bars right of 0 make the first set happier than the reference, bars left make it sadder. "
         "+↑ happier-than-reference word used more; −↓ sadder word used less (both raise the score); "
         "−↑ sadder word used more; +↓ happier word used less (both lower it).\n"
         "Dark bars = word used more often in the first set; light bars = used less often. Top 30 words by absolute contribution. "
         "Method: Dodds et al. (2011), Reagan et al. (2017).", fontsize=7.6, color="#52514e", va="bottom")
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
fig.savefig(FIG + "fig5_word_shift.png", dpi=200); fig.savefig(FIG + "fig5_word_shift.pdf"); print("ok")
