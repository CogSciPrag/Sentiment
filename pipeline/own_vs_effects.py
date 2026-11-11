"""Own Hedonometer score of each tested expression vs the mean sentiment of its effects (raw scores, average over the four corpora).
Own score: the labMT-en-v2 score of the expression as written if it is one listed word; for a multi-word expression the mean of the
scores of those of its words that labMT lists (mechanical, no judgement; 'n_words_rated' says how many of its words were rated).
Effects: per corpus the mean of the line scores (agg_score) of the lines that enter the results; the four corpora are averaged with equal weight
(a corpus counts if it has >= 30 scored lines); SE = sqrt(sum_c var_c/n_c)/k; Welch-Satterthwaite t test of (average - own score) = 0; Holm over all tests.
Usage: python own_vs_effects.py <outdir>"""
import sys, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "/home/claude/work/gr")
from causesent import config
from causesent.targets import DISPLAY
W = pd.read_csv("/mnt/user-data/uploads/sentiment/word_list_v6.tsv", sep="\t")
lex = pd.read_csv("/mnt/user-data/uploads/sentiment/lexicons/labMT-en-v2.tsv", sep="\t")
H = dict(zip(lex.word, lex.happs)); SD = dict(zip(lex.word, lex.stddev))
L = pd.read_pickle(config.RESULTS / "lines_raw.pkl"); L = L[L.agg_score.notna()]
ORDER = ["ententen21", "bnc", "coca", "reddit"]; MIN_N = 30
G = {k: g.agg_score.values for k, g in L.groupby(["corpus", "item"])}
rows = []
for _, r in W.iterrows():
    it = r["item"]; ws = [w for w in it.split() if w not in ("(of)", "(clause)", "X", "Y")]   # placeholders are not words
    rated = [(w, H[w]) for w in ws if w in H]
    own = np.mean([s for _, s in rated]) if len(rated) == len(ws) else np.nan   # own score only if labMT lists every word of the expression
    row = dict(item=DISPLAY.get(it, it), group=r["group"], own_score=own, own_sd_of_word=(SD[ws[0]] if len(ws) == 1 and ws[0] in H else np.nan),
               n_words=len(ws), n_words_rated=len(rated), words_rated=", ".join(f"{w} {s:.2f}" for w, s in rated),
               single_word=len(ws) == 1)
    ms, vs, ns, used = [], [], [], []
    for c in ORDER:
        d = G.get((c, it), np.array([]))
        row[f"mean_{c}"] = d.mean() if len(d) >= MIN_N else np.nan; row[f"n_{c}"] = len(d)
        if len(d) >= MIN_N: ms.append(d.mean()); vs.append(d.var(ddof=1)); ns.append(len(d)); used.append(c)
    k = len(ms); row["corpora_used"] = k
    M = np.mean(ms); se2 = np.array(vs) / np.array(ns); se = np.sqrt(se2.sum()) / k
    df = se2.sum() ** 2 / (se2 ** 2 / (np.array(ns) - 1)).sum()
    row.update(effects_mean=M, se=se, diff=M - own, ci_low=M - own - stats.t.ppf(.975, df) * se, ci_high=M - own + stats.t.ppf(.975, df) * se)
    row["p"] = 2 * stats.t.sf(abs(M - own) / se, df) if own == own else np.nan
    rows.append(row)
D = pd.DataFrame(rows)
m = D.p.notna(); p = D.loc[m, "p"].values; o = np.argsort(p); adj = np.empty(len(p)); run = 0
for rank, i in enumerate(o): run = max(run, (len(p) - rank) * p[i]); adj[i] = min(1, run)
D.loc[m, "p_holm"] = adj
D["stars"] = np.where(~m, "", np.select([D.p_holm < .001, D.p_holm < .01, D.p_holm < .05], ["***", "**", "*"], "n.s."))
D["direction"] = np.where(~m, "no own score", np.where(D.p_holm < .05, np.where(D["diff"] > 0, "effects happier than the word", "effects sadder than the word"), "n.s."))
D.to_csv(sys.argv[1] + "/own_score_vs_effects.tsv", sep="\t", index=False, float_format="%.4f")
pd.set_option("display.width", 250, "display.max_rows", 200)
print(D[["item", "own_score", "n_words_rated", "n_words", "effects_mean", "corpora_used", "diff", "p_holm", "stars"]].round(3).to_string())
print(D.direction.value_counts()); print(D.groupby("single_word").direction.value_counts())
