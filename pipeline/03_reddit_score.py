"""Step 3 (Reddit): parse the selected sentences, find each studied expression and its effect, and score it.

Input:  data/work/reddit_sentences_month/   (step 2 with --select study)
Output: data/work/reddit_scored/parts/*.tsv.gz   one row per occurrence of a studied expression (see causesent/frames.py
        for the columns), plus the sentence number and the first 300 characters of the sentence.
Resumable: one output file per chunk of 20,000 sentences; finished chunks are skipped.

Usage: python pipeline/03_reddit_score.py [n_processes]
"""
import glob, gzip, os, sys
import pandas as pd, spacy
from causesent import config, frames as F

src = config.WORK / "reddit_sentences_month"
outdir = config.WORK / "reddit_scored" / "parts"
nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 2
CHUNK = 20000
outdir.mkdir(parents=True, exist_ok=True)
nlp = spacy.load(config.SPACY_MODEL, disable=["ner"])

def read_block(fn):
    with gzip.open(fn, "rb") as fh:            # binary: sentences may contain \r or other line separators
        for raw in fh:
            idx, words_before, sent = raw.decode("utf-8")[:-1].split("\t", 2)
            yield int(idx), sent

for bfn in sorted(glob.glob(str(src / "sentences" / "block_*.tsv.gz"))):
    b = os.path.basename(bfn).split(".")[0]
    sents = list(read_block(bfn))
    for c0 in range(0, len(sents), CHUNK):
        out = outdir / f"{b}_{c0 // CHUNK:03d}.tsv.gz"
        if out.exists(): continue
        chunk = sents[c0:c0 + CHUNK]
        rows = []
        for (idx, s), doc in zip(chunk, nlp.pipe((s for _, s in chunk), batch_size=500, n_process=nproc)):
            for r in F.find_all(doc):
                r["sent_idx"] = idx; r["trigger"] = doc[r["trigger_i"]].text; r["sentence"] = s[:300]
                rows.append(r)
        pd.DataFrame(rows).to_csv(str(out) + ".tmp", sep="\t", index=False, compression="gzip")
        os.replace(str(out) + ".tmp", out)
        print(f"{out.name}: {len(chunk)} sentences, {len(rows)} records", flush=True)
print("done", flush=True)
