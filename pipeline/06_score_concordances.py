"""Step 6 (enTenTen21, BNC, COCA): score the concordance samples of the 80 studied expressions.

For each concordance line: rebuild the sentence around the search hit (left context, hit, right context, cut at the
sentence boundaries Sketch Engine marks with </s><s>), parse it with spaCy, and analyse only the searched expression
at the hit with causesent/frames.py. One record per line (or 'not matched' when the parser does not find it there).

  --source sketchengine   reads data/raw/sketchengine/se_batch_*.txt and se_*.csv (Sketch Engine CSV exports)
  --source coca           reads data/work/coca_prepared.txt (step 5)
Baseline samples ('BASELINE' in the sample name, or the file se_baseline_other_verbs.txt) are skipped here; they are
scored in steps 7 and 8.

Output: data/work/<source>_scored/<sample>.records.tsv, one file per corpus and expression. Existing files are skipped.
Usage:  python pipeline/06_score_concordances.py --source sketchengine
"""
import argparse, csv, glob, io, os, re
import pandas as pd, spacy
from causesent import config, frames as F
from causesent.targets import ITEMS

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--source", choices=["sketchengine", "coca"], required=True)
a = ap.parse_args()
if a.source == "sketchengine":
    d = config.RAW / "sketchengine"
    inputs = sorted(glob.glob(str(d / "se_batch_*.txt"))) + sorted(glob.glob(str(d / "se_*.csv")))
else:
    inputs = [str(config.WORK / "coca_prepared.txt")]
outdir = config.WORK / f"{a.source}_scored"
outdir.mkdir(parents=True, exist_ok=True)
nlp = spacy.load(config.SPACY_MODEL, disable=["ner"])

def samples(path):
    """(sample name, CSV text) for each sample in a file; batch files hold several samples after '### FILE' lines."""
    txt = open(path, encoding="utf-8-sig").read()
    if txt.startswith("### FILE "):
        for part in txt.split("### FILE ")[1:]:
            name, body = part.split("\n", 1)
            yield name.strip(), body
    else:
        yield os.path.basename(path), txt

def clean(s): return re.sub(r"\s+", " ", s.replace("</s>", " ").replace("<s>", " ")).strip()

for path in inputs:
    for name, body in samples(path):
        if "BASELINE" in name.upper(): continue
        out = outdir / name.replace(".csv", ".records.tsv")
        if out.exists(): continue
        rows = list(csv.reader(io.StringIO(body)))
        meta = {r[0]: r[1:] for r in rows[:8] if r and r[0] in ("corpus", "item", "query", "hits_in_corpus", "sample")}
        corpus = meta.get("corpus", [""])[0]; item = meta.get("item", [""])[0]
        if item not in ITEMS:   # older single-file export without an 'item' row: take the expression from the file name
            item = name.split("__", 1)[1].replace(".csv", "").replace("_", " ")
            item = {"because clause": "because (clause)"}.get(item, item)
        start = next(i for i, r in enumerate(rows) if r[:1] == ["Reference"]) + 1
        texts, spans = [], []
        for r in rows[start:]:
            if len(r) < 4: texts.append(""); spans.append((0, 0)); continue
            ref, left, kwic, right = r[:4]
            L = clean(left.split("<s>")[-1]) if "<s>" in left else clean(left)        # same sentence only
            R = clean(right.split("</s>")[0]) if "</s>" in right else clean(right)
            K = clean(kwic)
            text = (L + " " + K + " " + R).strip()
            k0 = len(L) + (1 if L else 0)                 # character span of the hit in the rebuilt sentence
            texts.append(text); spans.append((k0, k0 + len(K)))
        recs = []
        for li, (doc, (s0, s1)) in enumerate(zip(nlp.pipe(texts, batch_size=500), spans)):
            at_hit = [t for t in doc if s0 <= t.idx < s1]
            got = F.find_all(doc[:], only_items={item}) if at_hit else []
            got = [g for g in got if s0 <= doc[g["trigger_i"]].idx < s1][:1]
            if not got:
                recs.append(dict(item=item, frame="not matched", group="", line=li, sentence=texts[li][:300])); continue
            g = got[0]; g["line"] = li; g["sentence"] = texts[li][:300]; recs.append(g)
        D = pd.DataFrame(recs); D.insert(0, "corpus", corpus)
        D.to_csv(str(out) + ".tmp", sep="\t", index=False); os.replace(str(out) + ".tmp", out)
        print(f"{name}: {len(rows) - start} lines, matched {int((D.frame != 'not matched').sum())}", flush=True)
