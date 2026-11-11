"""Step 7 (enTenTen21, BNC): the baseline, i.e. all other lexical verbs, scored with the same rules.

Sample: 10,000 random concordance lines per corpus for the query [tag="VV.*"] (any lexical verb), downloaded from
Sketch Engine in the same format as the item samples. For each line the verb at the hit is analysed as a generic item,
'other verbs', with the verb rules of causesent/frames.py. Lines whose verb is the trigger lemma of a studied
expression are kept but marked 'target verb' (and left out of the results).

Since 1 Nov 2026 the same lines also give the baseline for the connectives (Glass's): every subordinating conjunction
(spaCy SCONJ, other than because / 'cause / cuz / cos) in the rebuilt sentence around the hit, scored on its main clause
as the generic item 'other conjunctions'. (Sketch Engine's tagset has no separate tag for subordinating conjunctions.)
Since 2 Nov 2026 they also give the NOUN baseline for the noun items (the cause of, result of, ...): every other noun in
'the N of X' or 'X is the N of Y' in the sentence, X scored, as the generic item 'other nouns' (frames._noun_base).

Input:  data/raw/sketchengine/se_baseline_other_verbs.txt
Output: data/work/sketchengine_baseline.records.tsv
Usage:  python pipeline/07_sketchengine_baseline.py
"""
import csv, io, re
import pandas as pd, spacy
from causesent import config, frames as F, targets

ITEM = "other verbs"
targets.ITEMS[ITEM] = ("*", "verb")          # a generic verb item: the same verb rules as cause, ensure, ...
CONJ = "other conjunctions"; targets.ITEMS[CONJ] = ("*", "sconj")
NOUN = "other nouns"; targets.ITEMS[NOUN] = ("*", "noun_base")     # noun baseline (decided 1 Nov 2026; added 2 Nov 2026)
TARGET_LEMMAS = {lem for lem, kind in list(targets.ITEMS.values()) if lem != "*"} - targets.BASELINE_KEEP
nlp = spacy.load(config.SPACY_MODEL, disable=["ner"])
def clean(s): return re.sub(r"\s+", " ", s.replace("</s>", " ").replace("<s>", " ")).strip()

config.ensure_dirs()
src = config.RAW / "sketchengine" / "se_baseline_other_verbs.txt"
out = config.WORK / "sketchengine_baseline.records.tsv"
recs = []
txt = open(src, encoding="utf-8-sig").read()
parts = [p.split("\n", 1) for p in txt.split("### FILE ")[1:]] if txt.startswith("### FILE ") else [("", txt)]
for name, body in parts:
    rows = list(csv.reader(io.StringIO(body)))
    corpus = next((r[1] for r in rows[:8] if r and r[0] == "corpus"), "")
    start = next(i for i, r in enumerate(rows) if r[:1] == ["Reference"]) + 1
    texts, spans = [], []
    for r in rows[start:]:
        if len(r) < 4: texts.append(""); spans.append((0, 0)); continue
        left, kwic, right = r[1:4]
        L = clean(left.split("<s>")[-1]) if "<s>" in left else clean(left)
        R = clean(right.split("</s>")[0]) if "</s>" in right else clean(right)
        K = clean(kwic); k0 = len(L) + (1 if L else 0)
        texts.append((L + " " + K + " " + R).strip()); spans.append((k0, k0 + len(K)))
    for li, (doc, (s0, s1)) in enumerate(zip(nlp.pipe(texts, batch_size=500), spans)):
        for c in doc:                         # connective baseline: the subordinating conjunctions in the sentence
            if c.pos_ == "SCONJ":
                for g in F.analyse(doc, CONJ, [c]):
                    g.update(corpus=corpus, line=li, sentence=texts[li][:300]); recs.append(g)
            elif c.pos_ == "NOUN":            # noun baseline: 'the N of X' / 'X is the N of Y' (X scored)
                for g in F.analyse(doc, NOUN, [c]):
                    g.update(corpus=corpus, line=li, sentence=texts[li][:300]); recs.append(g)
        verb = [t for t in doc if s0 <= t.idx < s1 and (t.pos_ == "VERB" or t.tag_.startswith("VB"))][:1]
        base = dict(corpus=corpus, item=ITEM, line=li, sentence=texts[li][:300])
        if not verb: recs.append(dict(base, frame="not matched")); continue
        t = verb[0]
        if F.lem(t) in TARGET_LEMMAS: recs.append(dict(base, frame="target verb", verb=F.lem(t))); continue
        got = F.analyse(doc, ITEM, [t])
        if not got: recs.append(dict(base, frame="not matched", verb=F.lem(t))); continue
        g = got[0]; g.update(corpus=corpus, line=li, sentence=texts[li][:300], verb=F.lem(t)); recs.append(g)
D = pd.DataFrame(recs); D.to_csv(out, sep="\t", index=False)
print(D.frame.value_counts().to_string())
