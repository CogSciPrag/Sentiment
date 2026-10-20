"""Parallel driver for glass_pipeline.py. Same outputs as get_reddit_data(), faster:
  1. The comment loop (filters, duplicate check, word budget, sent_tokenize, pmw counts) runs
     sequentially exactly as in her cell 59, so the same comments and sentences are selected.
  2. The per-sentence work (get_sconj, get_clauses, get_verb, scoring) is spread over worker
     processes; results are written back in the original order.
  3. nlp() is memoised on the input string: she parses the same sentence several times
     (get_sconj, get_clauses inside get_sconj, get_clauses again); spaCy's output for a string is
     deterministic, so re-using the parse changes nothing.
Usage: python glass_pipeline_fast.py INPUT.jsonl[.gz] OUTDIR [reddit_max_words] [n_workers]
"""
import csv, os, sys, string, functools
from collections import defaultdict
from multiprocessing import Pool
import glass_pipeline as G
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer

_raw_nlp = G.nlp
@functools.lru_cache(maxsize=256)
def _cached(text):
    return _raw_nlp(text)
class _NLP:
    def __call__(self, text): return _cached(text)
    def __getattr__(self, a): return getattr(_raw_nlp, a)
G.nlp = _NLP()   # her functions look up the global name `nlp` in module G

VF = ["verb", "trans_intrans", "clause", "subreddit", "clause_hedon", "dobj", "dobj_ont", "nsubj", "nsubj_ont",
      "ccomp", "ccomp_hedon", "nsubj_hedon", "dobj_hedon"]
CF = ["sentence", "subreddit", "sconj", "main_clause", "subor_clause", "pobj", "pobj_ont", "pobj_hedon",
      "main_clause_hedon", "subor_clause_hedon"]

def process_sentence(args):
    sent, sub = args
    conj_rows, verb_rows = [], []
    if len(sent.translate(str.maketrans('', '', string.punctuation)).split(" ")) >= 5:
        sconj_info = G.get_sconj(sent)
        if sconj_info:
            d0 = sconj_info; d0["subreddit"] = sub; d0["sentence"] = sent
            d0["main_clause_hedon"] = G.get_hedon_sent(sconj_info["main_clause"])
            d0["subor_clause_hedon"] = G.get_hedon_sent(sconj_info["subor_clause"])
            d0["pobj_hedon"] = G.get_hedon_word(sconj_info["pobj"])
            conj_rows.append(d0)
    for clause in G.get_clauses(sent):
        if len(clause.translate(str.maketrans('', '', string.punctuation)).split(" ")) >= 3:
            verb_info = G.get_verb(clause)
            if verb_info:
                d1 = verb_info; d1["subreddit"] = sub; d1["clause"] = clause
                d1["nsubj_hedon"] = G.get_hedon_word(verb_info["nsubj"])
                d1["dobj_hedon"] = G.get_hedon_word(verb_info["dobj"])
                d1["ccomp_hedon"] = G.get_hedon_sent(verb_info["ccomp"])
                d1["clause_hedon"] = G.get_hedon_sent(clause)
                verb_rows.append(d1)
    return conj_rows, verb_rows

def sentences(bigfile_name, reddit_max, pmw_dict, stats):
    """Her cell-59 comment loop, yielding (sentence, subreddit) instead of processing inline."""
    lemmatizer = WordNetLemmatizer()
    counter = 0; wordcount = 0; already_seen_comments = set()
    for d in G.get_reader(bigfile_name):
        if wordcount > reddit_max:
            break
        sub = d["subreddit"].lower()
        if (sub == "askreddit" and wordcount < reddit_max):
            d["body"] = d["body"].replace("&gt;", "").replace("\"", "").replace("*", "").replace("\\", "")
            if (not " bot " in d["body"]) and (not "[deleted]" in d["body"]) and (not "[removed]" in d["body"]) and \
               (not ("contact the moderators of this subreddit" in d["body"])) and not d["body"] in already_seen_comments:
                already_seen_comments.add(d["body"])
                wordcount += len(d["body"].split(" "))
                counter += 1
                stats["words"] = wordcount; stats["comments"] = counter; stats["last_id"] = d.get("id")
                for sent in sent_tokenize(d["body"].replace("\t", " ").replace("\n", " ").replace(".", ". ")):
                    for word in sent.split(" "):
                        pmw_dict[lemmatizer.lemmatize(word.lower())] += 1
                    yield (sent, sub)

def batched(gen, n):
    b = []
    for x in gen:
        b.append(x)
        if len(b) == n:
            yield b; b = []
    if b: yield b

def run(bigfile_name, outdir, reddit_max=10000000, workers=2):
    os.makedirs(outdir, exist_ok=True)
    verb_writer = csv.DictWriter(open(os.path.join(outdir, "reddit_verbclause_sentiment.txt"), "w"), VF, delimiter='\t')
    conj_writer = csv.DictWriter(open(os.path.join(outdir, "reddit_sconj_sentiment.txt"), "w"), CF, delimiter='\t')
    verb_writer.writeheader(); conj_writer.writeheader()
    pmw_dict = defaultdict(int); stats = {}
    nsent = 0
    with Pool(workers) as pool:
        for batch in batched(sentences(bigfile_name, reddit_max, pmw_dict, stats), 20000):
            for conj_rows, verb_rows in pool.imap(process_sentence, batch, chunksize=200):
                for r in conj_rows: conj_writer.writerow(r)
                for r in verb_rows: verb_writer.writerow(r)
            nsent += len(batch)
            print(f"{nsent} sentences, {stats['words']} words of {reddit_max} "
                  f"({100*stats['words']/reddit_max:.1f}%), last comment {stats.get('last_id')}", flush=True)
    print("DONE, total wordcount = " + str(stats.get("words")), f"comments = {stats.get('comments')}", flush=True)
    w = csv.DictWriter(open(os.path.join(outdir, "reddit_word_pmw.csv"), "w"), fieldnames=["word", "pmw"])
    w.writeheader()
    for word in pmw_dict:
        w.writerow({"word": word, "pmw": pmw_dict[word] * 1000000 / reddit_max})

if __name__ == "__main__":
    inp, outdir = sys.argv[1], sys.argv[2]
    rmax = int(sys.argv[3]) if len(sys.argv) > 3 else 10000000
    nw = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    run(inp, outdir, rmax, nw)
    G.postprocess(outdir)
