"""Step 4 (Reddit): the baseline, i.e. all other lexical verbs, scored with the same rules as the studied expressions.

Sample: a random 3-million-word subsample (fixed seed) of the first 10 million words of AskReddit, January 2018,
which is the text Glass's own baseline comes from. Every lexical verb in it that is not be/have/do and not the trigger
lemma of a studied expression is analysed as a generic item, 'other verbs', with the verb rules of causesent/frames.py.

Input:  data/work/reddit_sentences_10M/   (step 2 with --select all --max-words 10000000)
Output: data/work/reddit_baseline/parts/*.tsv.gz and sample_info.txt
Usage:  python pipeline/04_reddit_baseline.py [n_processes]   (BASE_WORDS=... changes the sample size)
"""
import glob, gzip, os, random, sys
import pandas as pd, spacy
from causesent import config, frames as F, targets

ITEM = "other verbs"
targets.ITEMS[ITEM] = ("*", "verb")              # a generic verb item: the same verb rules as cause, ensure, ...
EXCLUDE = {lem for lem, kind in list(targets.ITEMS.values()) if lem != "*"} | {"be", "have", "do"}
TARGET_WORDS, CHUNK, SEED = int(os.environ.get("BASE_WORDS", 3_000_000)), 20000, 20260930
nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 2
outdir = config.WORK / "reddit_baseline"
(outdir / "parts").mkdir(parents=True, exist_ok=True)

S = []
for fn in sorted(glob.glob(str(config.WORK / "reddit_sentences_10M" / "sentences" / "block_*.tsv.gz"))):
    for raw in gzip.open(fn, "rb"):
        idx, words_before, s = raw.decode("utf-8")[:-1].split("\t", 2); S.append((int(idx), s))
random.Random(SEED).shuffle(S)
sample, words = [], 0
for idx, s in S:
    if words >= TARGET_WORDS: break
    sample.append((idx, s)); words += len(s.split())
sample.sort()
open(outdir / "sample_info.txt", "w").write(f"sentences {len(sample)}\nwords {words}\nseed {SEED}\n")

nlp = spacy.load(config.SPACY_MODEL, disable=["ner"])
for c0 in range(0, len(sample), CHUNK):
    out = outdir / "parts" / f"part_{c0 // CHUNK:03d}.tsv.gz"
    if out.exists(): continue
    chunk = sample[c0:c0 + CHUNK]; rows = []
    for (idx, s), doc in zip(chunk, nlp.pipe((s for _, s in chunk), batch_size=500, n_process=nproc)):
        for t in doc:
            if t.pos_ != "VERB" or F.lem(t) in EXCLUDE: continue
            for r in F.analyse(doc, ITEM, [t]):
                r.update(verb=F.lem(t), sent_idx=idx, sentence=s[:300]); rows.append(r)
    pd.DataFrame(rows).to_csv(str(out) + ".tmp", sep="\t", index=False, compression="gzip"); os.replace(str(out) + ".tmp", out)
    print(f"part {c0 // CHUNK:03d}: {len(chunk)} sentences, {len(rows)} verb records", flush=True)
print("done", flush=True)
