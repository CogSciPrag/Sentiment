"""Whole-month check for panel A: Glass's own per-sentence code, UNPATCHED (her nlp calls, no memoisation, model
en_core_web_sm 3.6.0 as in her notebook), run on the month's target sentences, then her post-filters.
Compared with our fast pipeline's month output (p2_month). Resumable per chunk.
Usage: python glass_month_check.py PASS1_DIR OUTDIR [n_workers] [max_sentences]"""
import sys, os, gzip, csv, json, glob, io, string
from multiprocessing import Pool
import re
import glass_pipeline as G
from targets import forms
FOCUS = [l for l in os.environ.get("FOCUS", "").split(",") if l]
RXF = re.compile(r"\b(" + "|".join(f for l in FOCUS for f in forms(l)) + r")\b", re.I) if FOCUS else None

VF = ["verb", "trans_intrans", "clause", "subreddit", "clause_hedon", "dobj", "dobj_ont", "nsubj", "nsubj_ont",
      "ccomp", "ccomp_hedon", "nsubj_hedon", "dobj_hedon"]
def proc(args):
    sent, sub = args; rows = []
    for clause in G.get_clauses(sent):
        if len(clause.translate(str.maketrans('', '', string.punctuation)).split(" ")) >= 3:
            vi = G.get_verb(clause)
            if vi:
                d1 = vi; d1["subreddit"] = sub; d1["clause"] = clause
                d1["nsubj_hedon"] = G.get_hedon_word(vi["nsubj"]); d1["dobj_hedon"] = G.get_hedon_word(vi["dobj"])
                d1["ccomp_hedon"] = G.get_hedon_sent(vi["ccomp"]); d1["clause_hedon"] = G.get_hedon_sent(clause)
                rows.append(d1)
    return rows

if __name__ == "__main__":
    p1, outdir = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    maxs = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    parts = os.path.join(outdir, "parts"); os.makedirs(parts, exist_ok=True)
    CHUNK = 10_000; done_s = 0
    with Pool(workers) as pool:
        for bfn in sorted(glob.glob(os.path.join(p1, "sentences", "block_*.tsv.gz"))):
            b = os.path.basename(bfn).split(".")[0]
            sents = [(raw.decode("utf-8")[:-1].split("\t", 2)[2], "askreddit") for raw in gzip.open(bfn, "rb")]
            if RXF: sents = [x for x in sents if RXF.search(x[0])]
            for c0 in range(0, len(sents), CHUNK):
                if maxs and done_s >= maxs: break
                fn = os.path.join(parts, f"{b}_{c0 // CHUNK:03d}.verb.tsv")
                chunk = sents[c0:c0 + CHUNK][: (maxs - done_s) if maxs else None]
                done_s += len(chunk)
                if os.path.exists(fn): continue
                buf = io.StringIO(newline=""); w = csv.DictWriter(buf, VF, delimiter="\t")
                for rows in pool.imap(proc, chunk, chunksize=50):
                    for r in rows: w.writerow(r)
                open(fn + ".tmp", "w", newline="").write(buf.getvalue()); os.replace(fn + ".tmp", fn)
                print(f"{b}_{c0 // CHUNK:03d} done ({done_s:,} sentences)", flush=True)
    open(os.path.join(outdir, "DONE"), "w").write("done\n"); print("DONE", flush=True)
