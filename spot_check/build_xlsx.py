"""COCA spot check, step 2: the spreadsheet with the sampled lines (10 per word; 1,000 in the 2 Nov 2026 run), the words scored, and the manual check.

The manual check (verdicts.py: ok / partly / wrong, with the type of problem) was made by reading all 800 lines of the
sample drawn under the main rules (seed 20261019). It refers to the sample by position, so it is only meaningful for
that sample; with a rule switch on, the sample and its scored words change. Since 1 Nov 2026 the verdicts are matched
by LINE, not by position: verdicts_20261019.tsv lists the 800 checked lines (word, COCA reference, line text, the effect
lemmas then scored, verdict). A verdict is carried over only when the same line is in the new sample with the same
effect lemmas; otherwise the check is left empty (the note says when an old verdict exists but the effect changed).
Input: data/results/spot_check/ (sample_lines.py). Output: data/results/spot_check/coca_sample10_scored_words.xlsx
Usage: python spot_check/build_xlsx.py
"""
import pandas as pd
from pathlib import Path
from causesent import config
from causesent.targets import WORD_GROUPS
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.formatting.rule import ColorScaleRule, CellIsRule
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

G = config.RESULTS / "spot_check"
S = pd.read_csv(G / "sample10.tsv", sep="\t", index_col=0).rename(columns={"item": "word"})
RULES = (G / "rules.txt").read_text().strip() if (G / "rules.txt").exists() else ""
if "polarity" not in S: S["polarity"] = ""
# Manual checks, matched by LINE (item, COCA reference, line text) and carried over only with the same effect lemmas.
# verdicts_20261019.tsv: the 800-line check of 19 Oct 2026 (main rules); verdicts_20261102.tsv: first pass by Claude on
# 2 Nov 2026 of the lines of the 101-word run that the 19 Oct check did not cover (to be reviewed by Dean). A later check of
# the same line with the same effect words takes precedence.
VERDICTS = [(lab, f) for lab, f in [("19 Oct 2026", "verdicts_20261019.tsv"), ("2 Nov 2026", "verdicts_20261102.tsv")]
            if (Path(__file__).resolve().parent / f).exists()]
VK = [(lab, {(r.item, r.reference, r.line_text): r for r in
             pd.read_csv(Path(__file__).resolve().parent / f, sep="\t", keep_default_na=False).itertuples()})
      for lab, f in VERDICTS]
def carried(x):
    """(check, problem type, note) of the latest check of this line with the same effect lemmas."""
    nl = lambda v: v if isinstance(v, str) else ""
    key, lem = (x["word"], x["reference"], x["line_text"]), (nl(x.get("N_lemmas")), nl(x.get("V_lemmas")))
    other = None
    for lab, K in reversed(VK):
        o = K.get(key)
        if o is None: continue
        if lem == (o.N_lemmas, o.V_lemmas):
            note = o.note if lab == "19 Oct 2026" else ("first pass 2 Nov 2026 (Claude)" + (": " + o.note if o.note else ""))
            return (o.check, o.problem_type, note)
        other = other or (lab, o)
    if other:
        lab, o = other
        return ("", "", f"checked on {lab} ({o.check}) with other effect words ({o.N_lemmas} | {o.V_lemmas}): verdict not carried over")
    return ("", "", "")
order = pd.DataFrame([(w, g) for g, ws in WORD_GROUPS.items() for w in ws], columns=["item", "group"]); grp = dict(zip(order["item"], order["group"]))
L = pd.read_csv(G / "coca_lines_all.tsv", sep="\t", low_memory=False)
S[["check", "ptype", "note"]] = [carried(x) for _, x in S.iterrows()]
N_CARRIED = int(S.check.ne("").sum())
S["effect_side"] = ["left of the word" if e else "right of the word" for e in S.effect_left]
S["prev"] = ["yes" if p else "" for p in S.prevention]

wb = Workbook()
SHEET = f"Sample ({len(S)} lines)"
H = Font(bold=True, color="FFFFFF"); HF = PatternFill("solid", fgColor="2B4C7E")
WR = Alignment(wrap_text=True, vertical="top"); TOP = Alignment(vertical="top")
thin = Side(style="thin", color="D0D0D0")
fills = {"ok": "E6F4EA", "partly": "FFF4CE", "wrong": "FCE4E4"}

def header(ws, names, widths, height=None):
    for j, (h, w) in enumerate(zip(names, widths), 1):
        c = ws.cell(row=1, column=j, value=h); c.font = H; c.fill = HF
        c.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[get_column_letter(j)].width = w
    if height: ws.row_dimensions[1].height = height

# ---- Read me
ws = wb.active; ws.title = "Read me"
txt = [
 ("COCA spot check: which words were scored", "title"),
 ("For each of the " + str(S.word.nunique()) + " studied expressions with COCA lines: 10 random COCA lines among those that entered the results, with the word(s) scored and a manual check of each.", ""),
 ("", ""),
 ("How it was made", "h"),
 ("• Source lines: the COCA samples (random concordance lines per expression from english-corpora.org; sizes and queries in docs/queries.tsv; make / get combined from per-form samples).", ""),
 ("• Scored with pipeline steps 5 and 6 (causesent/frames.py, causesent/scoring.py), spaCy 3.6.0 with en_core_web_sm 3.6.0, Hedonometer " + config.LABMT.name + ".", ""),
 ("• Rules: " + (RULES or "(not recorded)") + ". Same rules as the results: main group only (passives and negated effects out), make/get causative frames only, with/for-N rule; raw Hedonometer scores.", ""),
 ("• Manual check: " + str(N_CARRIED) + " of the " + str(len(S)) + " lines have a verdict (same COCA line, same effect words): "
   + str(int(S.note.str.startswith("first pass 2 Nov").sum())) + " from Claude's first pass of 2 Nov 2026 (note 'first pass 2 Nov 2026'; to be reviewed in 'your check'), the others from the check of 19 Oct 2026. "
   "Empty for lines without a verdict.", ""),
 ("• Sample: 10 lines per word, drawn at random (numpy default_rng, seed 20261019) from the lines that received a score.", ""),
 ("• The manual check of 19 Oct 2026 was made on the sample drawn under the main rules; its verdicts (verdicts_20261019.tsv) are matched to this sample line by line.", ""),
 ("", ""),
 ("Columns in '" + SHEET + "'", "h"),
 ("• line: the KWIC line, searched word in [[double brackets]]; ' | ' marks a sentence or paragraph break or a COCA copyright gap.", ""),
 ("• frame: the frame the parser assigned. polarity: positive or negative form of the studied word ('didn't cause', 'nothing caused' = negative). "
   "N scored / V scored: the words with their Hedonometer score. With the form-first rule: 'problems (2.92)' = the word form was scored; "
   "'ran -> run (5.24)' = the form is not in the Hedonometer, so its lemma was scored; '(–)' = neither is in the Hedonometer, so not scored.", ""),
 ("• line score: N alone, or the mean of N and V when there is a complement. Coloured red to green, centred at 5.", ""),
 ("• effect side: for result from, because of etc. the effect is on the left. prevention = yes: the effect is what does NOT happen.", ""),
 ("• manual check: ok / partly (main word defensible, but part of the effect missed or an extra word scored) / wrong (the scored word is not the effect, or the line should not count). 'problem type' and 'note' explain each flag; see the 'Problem types' sheet.", ""),
 ("• your check / your note: empty, with an ok / partly / wrong drop-down, for your own pass.", ""),
 ("", ""),
 ("Deliberately not flagged", "h"),
 ("• Negation of the causal word itself ('does not require', 'doesn't make a difference to', 'nobody forced me'): kept in the main results, as in Glass (2023), and marked in the polarity column.", ""),
 ("• The causee as N ('caused me to question', 'forced us to act'): the agreed N + V scoring rule.", ""),
 ("", ""),
 ("Other sheets", "h"),
 ("• Summary by word: COCA lines, scored lines, and the check counts per word. • Lines left out: why each word's other lines did not enter the results. • Problem types: what each flag means, with counts.", ""),
 ("• Every scored COCA line (scored or not, with the reason) is in coca_lines_all.tsv next to this file.", ""),
]
for i, (t, kind) in enumerate(txt, 1):
    c = ws.cell(row=i, column=1, value=t); c.alignment = Alignment(wrap_text=True, vertical="top")
    if kind == "title": c.font = Font(bold=True, size=14)
    elif kind == "h": c.font = Font(bold=True, size=11)
ws.column_dimensions["A"].width = 140

# ---- Sample
ws = wb.create_sheet(SHEET)
names = ["#", "word", "group", "COCA reference", "line", "effect side", "prevention", "frame", "polarity", "N scored", "V scored",
         "line score", "manual check", "problem type", "note", "your check", "your note"]
header(ws, names, [5, 16, 22, 24, 80, 11, 10, 20, 10, 26, 22, 9, 10, 15, 48, 10, 30], 32)
r = 2
for i, x in S.iterrows():
    vals = [i + 1, x["word"], grp.get(x["word"], ""), x["reference"], x["line_text"], x["effect_side"], x["prev"], x["frame"],
            x["polarity"] if isinstance(x["polarity"], str) else "", x["N_scored"] if isinstance(x["N_scored"], str) else "", x["V_scored"] if isinstance(x["V_scored"], str) else "",
            round(float(x["line_score"]), 2), x["check"], x["ptype"], x["note"], None, None]
    for j, v in enumerate(vals, 1):
        c = ws.cell(row=r, column=j, value=v); c.alignment = WR if j in (5, 10, 11, 15, 17) else TOP; c.border = Border(bottom=thin)
    if x["check"] in fills: ws.cell(row=r, column=13).fill = PatternFill("solid", fgColor=fills[x["check"]])
    r += 1
n = r - 1
ws.freeze_panes = "C2"; ws.auto_filter.ref = f"A1:Q{n}"
ws.conditional_formatting.add(f"L2:L{n}", ColorScaleRule(start_type="num", start_value=1, start_color="D7301F", mid_type="num",
                              mid_value=5, mid_color="FFFFFF", end_type="num", end_value=9, end_color="1A9850"))
dv = DataValidation(type="list", formula1='"ok,partly,wrong"', allow_blank=True); ws.add_data_validation(dv); dv.add(f"P2:P{n}")
for k, col in fills.items():
    ws.conditional_formatting.add(f"P2:P{n}", CellIsRule(operator="equal", formula=[f'"{k}"'], fill=PatternFill("solid", fgColor=col)))

# ---- Summary by word
ws = wb.create_sheet("Summary by word")
fate = L.groupby(["item", "fate"]).size().unstack(fill_value=0)
header(ws, ["word", "group", "COCA lines", "scored lines (in results)", "sampled", "ok", "partly", "wrong", "problem types in the sample"],
       [18, 32, 9, 12, 8, 6, 7, 7, 55], 32)
r = 2
for it in order["item"]:
    if it not in fate.index: continue      # no COCA lines for this word ('as' is left out of COCA by design, 2 Nov 2026)
    g = S[S.word == it]; f = fate.loc[it]
    pt = g[g.check.isin(["partly", "wrong"])].ptype.value_counts()
    vals = [it, grp[it], int(f.sum()), int(f.get("scored", 0)), len(g), int((g.check == "ok").sum()), int((g.check == "partly").sum()),
            int((g.check == "wrong").sum()), ", ".join(f"{k} ({v})" for k, v in pt.items())]
    for j, v in enumerate(vals, 1): ws.cell(row=r, column=j, value=v).alignment = WR if j == 9 else TOP
    r += 1
ws.cell(row=r, column=1, value="Total").font = Font(bold=True)
for j, col in zip(range(3, 9), "CDEFGH"):
    ws.cell(row=r, column=j, value=f"=SUM({col}2:{col}{r - 1})").font = Font(bold=True)
ws.freeze_panes = "B2"; ws.auto_filter.ref = f"A1:I{r - 1}"
ws.conditional_formatting.add(f"H2:H{r - 1}", ColorScaleRule(start_type="num", start_value=0, start_color="FFFFFF",
                              end_type="num", end_value=6, end_color="F4A3A3"))

# ---- Lines left out
ws = wb.create_sheet("Lines left out")
fc = [c for c in fate.columns if c != "scored"]
header(ws, ["word", "COCA lines", "scored"] + [c.replace("excluded: ", "") for c in fc], [18, 9, 9] + [14] * len(fc), 75)
for r, it in enumerate([i for i in order["item"] if i in fate.index], 2):     # words with COCA lines only
    f = fate.loc[it]
    for j, v in enumerate([it, int(f.sum()), int(f.get("scored", 0))] + [int(f[c]) for c in fc], 1): ws.cell(row=r, column=j, value=v)
ws.freeze_panes = "B2"

# ---- Problem types
desc = {
 "partitive": "The head noun is a quantity or part word ('a lot of', 'part of', '72 percent of', 'a kind of', 'some of', 'more than'), so the quantity is scored instead of the thing ('kind' scores 7.24, 'lot' 5.78).",
 "number": "A number word is scored ('$1 billion' -> billion 7.56).",
 "parse": "Parser error: the wrong word is picked as subject, object or complement.",
 "sense": "The word is used in a non-causal sense (contribute to = write for or donate; render = depict; oblige = do a favour; secure = fasten; affect = put on; save = store), or the scored word is read in the wrong sense.",
 "other clause": "A verb from a neighbouring clause or sentence is scored as the effect.",
 "relative clause": "The noun a relative clause modifies is scored ('the guy that doesn't help' -> guy).",
 "not the item": "The hit is not the studied expression ('cause' = 'cos; 'thanks' = gratitude; adverbial 'enough').",
 "negated effect": "Negation inside the effect ('led to no good', 'yielded zero studies'): should go to the negated group under the 30 Sep rule, but the code only checks negation on a complement verb.",
 "ditransitive": "'cause/guarantee someone something': the indirect object is scored instead of the direct object or clause.",
 "idiom": "Idioms ('force his way', 'dying to hear', 'don't get me wrong', 'let me know').",
 "text": "Text artefacts (hyphenated syllables; \"Ive's\" read as I've).",
 "not causative": "get / let not in a causative use ('got the decal', 'He got me', 'Let's push').",
 "purpose clause": "A purpose clause ('stopped to consider', 'to prevent panic') scored as the effect.",
 "missed passive": "Passive not detected.",
 "into-phrase": "'force / provoke N into V-ing / into N': the into-phrase (the effect) is not captured.",
 "missed part": "Part of the effect missed (predicate adjective, a to-V after a PP, etc.).",
}
ws = wb.create_sheet("Problem types")
header(ws, ["problem type", "wrong", "partly", "what it is", "examples (# in the sample)"], [16, 8, 8, 95, 32])
P = S[S.check.isin(["partly", "wrong"])].groupby("ptype").check.value_counts().unstack(fill_value=0)
P["k"] = P.get("wrong", 0) * 100 + P.get("partly", 0)
for r, (pt, row) in enumerate(P.sort_values("k", ascending=False).iterrows(), 2):
    ex = ", ".join(str(i + 1) for i in S[S.ptype == pt].index[:10])
    for j, v in enumerate([pt, int(row.get("wrong", 0)), int(row.get("partly", 0)), desc.get(pt, ""), ex], 1):
        ws.cell(row=r, column=j, value=v).alignment = WR
out = G / "coca_sample10_scored_words.xlsx"; wb.save(out); print(out)
