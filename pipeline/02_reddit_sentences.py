"""Step 2 (Reddit): select AskReddit comments and split them into sentences, exactly as Glass (2023) did.

This reproduces Glass's comment loop (her notebook GetCausationClausesFromReddit_updated2023.ipynb, cell 59):
the same comment filters (bots, [deleted], [removed], moderator notices), the same duplicate check, the same text
clean-up and NLTK sentence splitting, the same running word count and per-million word counts. Instead of parsing
each sentence straight away, it saves the selected sentences for the next step.

Which sentences are saved (--select):
  study  sentences containing a word form of one of the 80 studied expressions (causesent/targets.py)
  glass  sentences containing one of the verbs of the Glass replication (glass_replication/targets.py)
  all    every sentence (used with --max-words 10000000 for the baseline and the 10-million-word replication)

Output folder data/work/<name>/: sentences/block_#####.tsv.gz (sentence number, words before it, sentence),
word_counts.tsv (lemma counts, as in Glass), pass1_stats.txt. The run is resumable: it checkpoints every 100,000
comments, so it can be restarted after an interruption.

Examples:
  python pipeline/02_reddit_sentences.py reddit_sentences_month --select study
  python pipeline/02_reddit_sentences.py reddit_sentences_10M --select all --max-words 10000000
"""
import argparse, gzip, hashlib, importlib.util, itertools, json, os, pickle, sys
from collections import defaultdict
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from causesent import config

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("name", help="output folder name under data/work/")
ap.add_argument("--select", choices=["study", "glass", "all"], required=True)
ap.add_argument("--max-words", type=int, default=0, help="Glass's word budget (reddit_max); 0 = whole input")
ap.add_argument("inputs", nargs="*", help="AskReddit JSONL.gz file(s); default: the output of step 1")
a = ap.parse_args()
inputs = a.inputs or [str(config.WORK / "reddit" / "askreddit_2018-01.jsonl.gz")]
if a.select == "study":
    from causesent.targets import has_target
elif a.select == "glass":
    spec = importlib.util.spec_from_file_location("glass_targets", config.REPO / "glass_replication" / "targets.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); has_target = m.has_target

outdir = config.WORK / a.name
BLOCK = 100_000
os.makedirs(outdir / "sentences", exist_ok=True)
ckpt = outdir / "checkpoint.pkl"
if (outdir / "pass1_stats.txt").exists():
    print("already done"); sys.exit(0)
if ckpt.exists():
    S = pickle.load(open(ckpt, "rb"))
    print(f"resuming after raw line {S['raw']:,}, {S['comments']:,} comments, {S['words']:,} words", flush=True)
else:
    S = dict(raw=0, comments=0, words=0, sent=0, target=0, block=0, seen=set(), counts=defaultdict(int), done=False)

lemmatizer = WordNetLemmatizer(); lem_cache = {}
# Glass keeps the full comment texts to detect duplicates; a 64-bit hash gives the same result with far less memory
def h64(s): return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "surrogatepass"), digest_size=8).digest(), "big")

def all_lines():
    for fn in inputs:
        for line in gzip.open(fn, "rt"):
            yield line

buf = []
def flush_block():
    fn = outdir / "sentences" / f"block_{S['block']:05d}.tsv.gz"
    with gzip.open(str(fn) + ".tmp", "wt") as f:
        f.writelines(buf)
    os.replace(str(fn) + ".tmp", fn)
    S["block"] += 1; buf.clear()
    pickle.dump(S, open(str(ckpt) + ".tmp", "wb"), protocol=4); os.replace(str(ckpt) + ".tmp", ckpt)

in_block = 0
for line in itertools.islice(all_lines(), S["raw"], None):
    S["raw"] += 1; in_block += 1
    d = json.loads(line)
    if a.max_words and S["words"] > a.max_words:          # Glass: `if wordcount > reddit_max: break`
        break
    if d["subreddit"].lower() == "askreddit" and (not a.max_words or S["words"] < a.max_words):
        body = d["body"].replace("&gt;", "").replace("\"", "").replace("*", "").replace("\\", "")
        if (" bot " not in body) and ("[deleted]" not in body) and ("[removed]" not in body) and \
           ("contact the moderators of this subreddit" not in body):
            h = h64(body)
            if h not in S["seen"]:
                S["seen"].add(h)
                words_before = S["words"]
                S["words"] += len(body.split(" ")); S["comments"] += 1
                for sent in sent_tokenize(body.replace("\t", " ").replace("\n", " ").replace(".", ". ")):
                    S["sent"] += 1
                    for word in sent.split(" "):                      # Glass's per-million word counts
                        w = word.lower(); lw = lem_cache.get(w)
                        if lw is None:
                            lw = lemmatizer.lemmatize(w)
                            if len(lem_cache) < 3_000_000: lem_cache[w] = lw
                        S["counts"][lw] += 1
                    if a.select == "all" or has_target(sent):
                        S["target"] += 1
                        buf.append(f"{S['sent']}\t{words_before}\t{sent}\n")
    if in_block >= BLOCK:
        flush_block(); in_block = 0
        print(f"{S['raw']:,} lines, {S['comments']:,} comments, {S['words']:,} words, "
              f"{S['target']:,}/{S['sent']:,} sentences saved", flush=True)
flush_block()
with open(outdir / "word_counts.tsv", "w") as f:
    for w, c in S["counts"].items():
        f.write(f"{w}\t{c}\n")
info = dict(raw_lines=S["raw"], comments_kept=S["comments"], words=S["words"], sentences=S["sent"],
            saved_sentences=S["target"], blocks=S["block"], select=a.select, max_words=a.max_words)
open(outdir / "pass1_stats.txt", "w").write(json.dumps(info) + "\n")
print("done", info, flush=True)
