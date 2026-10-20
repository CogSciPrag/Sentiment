"""Pass 2: run Glass's per-sentence extraction (get_sconj, get_clauses, get_verb and scoring, unchanged, via
glass_pipeline_fast.process_sentence) on the sentences saved by pass 1, then her post-filters (postprocess).
pmw = count * 1e6 / total words of the pass-1 sample (her formula, with reddit_max = sample size).
Resumable: results are written per chunk of CHUNK sentences; finished chunks are skipped on restart.
Usage: python glass_pass2_targets.py PASS1_DIR OUTDIR [n_workers]
"""
import sys, os, gzip, csv, json, glob, io
from multiprocessing import Pool
import glass_pipeline_fast as F      # patches her nlp with the memoised version
import glass_pipeline as G

p1, outdir = sys.argv[1], sys.argv[2]
workers = int(sys.argv[3]) if len(sys.argv) > 3 else 2
CHUNK = 10_000
stats = json.loads(open(os.path.join(p1, "pass1_stats.txt")).read())
parts = os.path.join(outdir, "parts"); os.makedirs(parts, exist_ok=True)

def read_block(fn):
    # binary read, split on \n only: sentences may contain \r, which text mode would treat as a line end
    with gzip.open(fn, "rb") as fh:
        for raw in fh:
            idx, wb, sent = raw.decode("utf-8")[:-1].split("\t", 2)
            yield (sent, "askreddit")

blocks = sorted(glob.glob(os.path.join(p1, "sentences", "block_*.tsv.gz")))
total = 0
with Pool(workers) as pool:
    for bfn in blocks:
        b = os.path.basename(bfn).split(".")[0]
        sents = list(read_block(bfn))
        for c0 in range(0, len(sents), CHUNK):
            tag = f"{b}_{c0 // CHUNK:03d}"
            vfn, cfn = os.path.join(parts, tag + ".verb.tsv"), os.path.join(parts, tag + ".sconj.tsv")
            if os.path.exists(vfn) and os.path.exists(cfn):
                continue
            vbuf, cbuf = io.StringIO(newline=""), io.StringIO(newline="")
            vw = csv.DictWriter(vbuf, F.VF, delimiter="\t"); cw = csv.DictWriter(cbuf, F.CF, delimiter="\t")
            for conj_rows, verb_rows in pool.imap(F.process_sentence, sents[c0:c0 + CHUNK], chunksize=100):
                for r in conj_rows: cw.writerow(r)
                for r in verb_rows: vw.writerow(r)
            for fn, bf in ((cfn, cbuf), (vfn, vbuf)):
                open(fn + ".tmp", "w", newline="").write(bf.getvalue()); os.replace(fn + ".tmp", fn)
            total += min(CHUNK, len(sents) - c0)
            print(f"{tag} done ({total:,} sentences this session)", flush=True)

# assemble in order, with headers, exactly like her output files
tags = sorted(os.path.basename(f)[:-9] for f in glob.glob(os.path.join(parts, "*.verb.tsv")))
for name, fields, suffix in (("reddit_verbclause_sentiment.txt", F.VF, ".verb.tsv"), ("reddit_sconj_sentiment.txt", F.CF, ".sconj.tsv")):
    with open(os.path.join(outdir, name), "w", newline="") as out:
        csv.DictWriter(out, fields, delimiter="\t").writeheader()
        for t in tags:
            out.write(open(os.path.join(parts, t + suffix), newline="").read())
with open(os.path.join(outdir, "reddit_word_pmw.csv"), "w") as f:
    w = csv.writer(f); w.writerow(["word", "pmw"])
    for raw in open(os.path.join(p1, "word_counts.tsv"), "rb"):
        word, c = raw.decode("utf-8")[:-1].rsplit("\t", 1)
        w.writerow([word, int(c) * 1_000_000 / (stats["max_words"] or stats["words"])])
G.postprocess(outdir)
print("DONE", flush=True)
