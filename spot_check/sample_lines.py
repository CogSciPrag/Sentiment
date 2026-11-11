"""COCA spot check, step 1: every COCA line with its fate (scored, or why it was left out), and a random sample of
10 scored lines per expression, showing which words were scored (raw Hedonometer scores) and the line's polarity.
Scored words follow the active rules (form-first rule, 1 Nov 2026: 'problems (2.92)' = the form was scored;
'ran -> run (5.24)' = the form is not in the Hedonometer, so its lemma was scored; 'xyz (–)' = neither, not scored).

Inputs: data/work/coca_prepared.txt (step 5) and data/work/coca_scored/ (step 6).
Outputs (data/results/spot_check/): coca_lines_all.tsv (contains corpus text: do not redistribute), sample10.tsv,
fates.tsv. Sampling: numpy default_rng(SEED) per expression, among the lines that received a score.
Usage: python spot_check/sample_lines.py
"""
import glob, os, csv, io, re, sys
import numpy as np, pandas as pd
from causesent import config, scoring as SL
from causesent.targets import WORD_GROUPS

SEED = 20261019
OUT = config.RESULTS / "spot_check"; OUT.mkdir(parents=True, exist_ok=True)

# 1. original KWIC rows (reference, left, kwic, right) per file
kwic = {}
for part in open(config.WORK / "coca_prepared.txt", encoding="utf-8").read().split("### FILE ")[1:]:
    name, body = part.split("\n", 1)
    if "BASELINE" in name: continue
    rows = list(csv.reader(io.StringIO(body)))
    start = next(i for i, r in enumerate(rows) if r[:1] == ["Reference"]) + 1
    kwic[name.strip().replace(".csv", "")] = rows[start:]

# 2. records, with their file name
recs = []
for f in sorted(glob.glob(str(config.WORK / "coca_scored" / "*.records.tsv"))):
    d = pd.read_csv(f, sep="\t", low_memory=False); d["file"] = os.path.basename(f).replace(".records.tsv", ""); recs.append(d)
T0 = pd.concat(recs, ignore_index=True)
T0["rid"] = np.arange(len(T0))

# 3. raw scoring + the aggregate rules (same functions as the saved results)
R = SL.rescore(T0, "raw")
P = SL.prep(R)                       # lines that enter the aggregate (score may still be NaN if no word is in the lexicon)
kept = set(P.rid[P.agg_score.notna()])
R = R.merge(P[["rid", "frame", "agg_score", "score_N", "score_NV"]].rename(columns={"frame": "frame_used", "agg_score": "agg_used",
            "score_N": "score_N_used", "score_NV": "score_NV_used"}), on="rid", how="left")

def fate(r):
    if r.rid in kept: return "scored"
    if r.frame == "not matched": return "excluded: parser did not find the item at the KWIC"
    if r.frame == "no effect": return "excluded: no effect found"
    if r.group == "passive": return "excluded: passive"
    if r.group == "negated": return "excluded: negated effect"
    if r.group != "main": return f"excluded: {r.group}"
    if r.frame == "target verb": return "excluded: target verb"
    if r.item in SL.AGG_FRAMES and r.frame not in SL.AGG_FRAMES[r.item]: return f"excluded: non-causative frame ({r.frame})"
    if pd.notna(r.frame_used): return "excluded: no scored word in the Hedonometer"
    return "excluded: other"
R["fate"] = [fate(r) for r in R.itertuples()]

H = SL.H
def words_scored(ws, fs=None):
    """The words of one slot as scored: lemmas ws with their forms fs (same order), following SL.keys()."""
    if not isinstance(ws, str) or not ws.strip(): return ""
    lem = ws.split(); key = SL.keys(ws, fs, H)
    frm = fs.split() if isinstance(fs, str) and len(fs.split()) == len(lem) else lem
    out = []
    if not config.FORM_FIRST: frm = lem          # lemma-only rule: show the lemmas, as in the 19 Oct 2026 check
    for f, l, k in zip(frm, lem, key):
        if k not in H: out.append(f"{f} (–)")
        elif k == f: out.append(f"{f} ({H[k]:.2f})")
        else: out.append(f"{f} -> {k} ({H[k]:.2f})")
    return ", ".join(out)
col = lambda c: R[c] if c in R else pd.Series(None, index=R.index, dtype=object)

def line_text(r):
    rows = kwic[r.file]; li = int(r.line)
    if li >= len(rows) or len(rows[li]) < 4: return "", r.sentence
    ref, left, k, right = rows[li][:4]
    cl = lambda s: re.sub(r"\s+", " ", s.replace("</s><s>", " | ").replace("</s>", " | ").replace("<s>", " ")).strip()
    return ref, f"{cl(left)} [[{cl(k)}]] {cl(right)}"

R[["reference", "line_text"]] = [line_text(r) for r in R.itertuples()]
R["N_scored"] = [words_scored(w, f) for w, f in zip(R.N_words, col("N_forms"))]
R["V_scored"] = [words_scored(w, f) for w, f in zip(R.V_words, col("V_forms"))]
if "polarity" not in R: R["polarity"] = ""
R["N_lemmas"] = R.N_words.fillna(""); R["V_lemmas"] = R.V_words.fillna("")   # effect words, to match old verdicts
# with/for-N rule: the direct object replaces N (prep does this); show it
wf = (R.frame == "V + N + with/for N") & ~R.item.isin(["blame", "credit"]) & R.rid.isin(kept)
R.loc[wf, "N_lemmas"] = R.loc[wf, "obj_words"].fillna("")
R.loc[wf, "N_scored"] = [words_scored(w, f) + "  [direct object used: with/for-N rule]"
                         for w, f in zip(R.loc[wf, "obj_words"], col("obj_forms")[wf])]
R["line_score"] = R.agg_used.where(R.rid.isin(kept))

order = [w for ws in WORD_GROUPS.values() for w in ws]
rows = []
for item in order:
    g = R[(R.item == item) & R.rid.isin(kept)]
    rng = np.random.default_rng(SEED)
    take = g.iloc[np.sort(rng.choice(len(g), size=min(10, len(g)), replace=False))] if len(g) else g
    rows.append(take)
S = pd.concat(rows)
cols = ["item", "reference", "line_text", "frame", "polarity", "N_scored", "V_scored", "line_score", "prevention", "effect_left",
        "N_lemmas", "V_lemmas"]
S[cols].reset_index(drop=True).to_csv(OUT / "sample10.tsv", sep="\t")
R[[c for c in ["item", "file", "line", "reference", "line_text", "frame", "group", "polarity", "fate", "N_words", "N_forms",
   "V_words", "V_forms", "obj_words", "N_mods", "N_scored", "V_scored", "score_N", "score_V", "line_score"] if c in R]].to_csv(OUT / "coca_lines_all.tsv", sep="\t", index=False, float_format="%.4f")
F = R.groupby(["item", "fate"]).size().unstack(fill_value=0)
F.to_csv(OUT / "fates.tsv", sep="\t")
(OUT / "rules.txt").write_text(config.rules_label() + "\n")     # build_xlsx.py uses this to decide on the old verdicts
print(len(S), "sample rows;", len(R), "lines;", R.fate.eq("scored").sum(), "scored |", config.rules_label())
