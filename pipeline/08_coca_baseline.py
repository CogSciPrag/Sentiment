"""Step 8 (COCA): the baseline, i.e. all other lexical verbs, scored with the same rules.

english-corpora.org offers no random sample of all verbs (a part-of-speech query lists word forms only), so the
baseline is built from random concordance samples of 11 frequent function words ('anchors': in, also, after, when,
would, about, then, where, could, into, over; 500 lines each). Every lexical verb in those lines (left context,
anchor, right context, within one paragraph and outside copyright gaps) is analysed with the same rules as the
studied expressions, as the generic item 'other verbs'. Excluded: be/have/do, the trigger lemmas of the studied
expressions, and verbs with fewer than 4 tokens to their right (their complement may be cut off; the items' own lines
have about 10 words of right context).

Input:  data/work/coca_prepared.txt (step 5; the samples whose name contains 'BASELINE')
Output: data/work/coca_baseline.records.tsv
Usage:  python pipeline/08_coca_baseline.py
"""
import csv, io, re
import pandas as pd, spacy
from causesent import config, frames as F, targets

ITEM = "other verbs"
targets.ITEMS[ITEM] = ("*", "verb")
TARGET_LEMMAS = {lem for lem, kind in list(targets.ITEMS.values()) if lem != "*"} | {"be", "have", "do"}
nlp = spacy.load(config.SPACY_MODEL, disable=["ner"])
def clean(s): return re.sub(r"\s+", " ", s).strip()

src = config.WORK / "coca_prepared.txt"
out = config.WORK / "coca_baseline.records.tsv"
recs = []
for part in open(src, encoding="utf-8").read().split("### FILE ")[1:]:
    name, body = part.split("\n", 1)
    if "BASELINE" not in name: continue
    rows = list(csv.reader(io.StringIO(body)))
    start = next(i for i, r in enumerate(rows) if r[:1] == ["Reference"]) + 1
    texts = []
    for r in rows[start:]:
        if len(r) < 4: continue
        full = clean(r[1] + " " + r[2] + " " + r[3])
        for seg in full.split("</s><s>"):          # each paragraph / gap-free stretch on its own
            seg = clean(seg)
            if len(seg.split()) >= 6: texts.append((name.strip(), seg))
    for (nm, s), doc in zip(texts, nlp.pipe((s for _, s in texts), batch_size=500)):
        for t in doc:
            if not (t.pos_ == "VERB" and t.tag_.startswith("VB")): continue
            if len(doc) - t.i - 1 < 4: continue
            lem = F.lem(t)
            if lem in TARGET_LEMMAS: continue
            got = F.analyse(doc, ITEM, [t])
            if not got: recs.append(dict(corpus="coca", item=ITEM, frame="not matched", verb=lem, anchor=nm)); continue
            g = got[0]; g.update(corpus="coca", verb=lem, anchor=nm, sentence=s[:300]); recs.append(g)
D = pd.DataFrame(recs); D.to_csv(out, sep="\t", index=False)
print(len(D), "verb tokens"); print(D.frame.value_counts().head(15).to_string())
