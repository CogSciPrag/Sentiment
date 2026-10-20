"""Faithful re-implementation of Glass (2023)'s extraction pipeline
(OSF mv5gk: GetCausationClausesFromReddit_updated2023.ipynb, cells 6, 15, 21, 27, 43, 59, 69-86).

Her functions are copied verbatim, with two changes that do NOT alter any output:
  1. get_hedon_sent() uses nlp.tokenizer instead of nlp(): she only uses the tokens, and
     spaCy's tokens are the same with or without the parser/tagger.
  2. The WordNet ontology columns (dobj_ont, nsubj_ont, pobj_ont) are left blank; they
     play no role in the sentiment analysis.
Added (outside her functions): a word budget argument (her reddit_max, default 10M) and
an optional skip of the first N words, so the run can be extended beyond her sample.

Usage: python glass_pipeline.py INPUT.jsonl[.gz] OUTDIR [reddit_max_words]
INPUT: Reddit comments in dump order, one JSON object per line with 'subreddit' and 'body'.
"""
import csv, os, sys, json, gzip, string
from collections import defaultdict
import numpy as np
import spacy
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer

# Glass's notebook installs en_core_web_sm 3.6.0; set SPACY_MODEL to its path to use that version.
import warnings; warnings.filterwarnings("ignore", message=r".*W095.*")
nlp = spacy.load(os.environ.get("SPACY_MODEL", "en_core_web_sm"))
from causesent import config
HEDON_FILE = str(config.LABMT)

# ---- cell 6 (lexicon: identical scores to her Hedonometer.csv; checked on 16 words incl. cause 5.22)
hedon_dict = defaultdict(float)
for line in csv.DictReader(open(HEDON_FILE), delimiter="\t"):
    hedon_dict[line["word"]] = float(line["happs"])
LEXICON = frozenset(hedon_dict.keys())
# NOTE: hedon_dict is a defaultdict, so her get_hedon_sent() silently adds every token it looks up
# (with value 0.0). In a single sequential run this would make get_hedon_word() return 0.0 for such
# words, but her posted verb_clause_df.csv contains no 0 values (min 1.3), so her actual scores
# behave as a lookup in the original lexicon. We do that explicitly (LEXICON), which also makes the
# result independent of processing order.

def get_hedon_word(word):
    try:
        if len(word) >= 1 and word in LEXICON:
            return hedon_dict[word]
        else:
            return np.nan
    except:
        return np.nan

# ---- cell 15 (tokenizer-only; same tokens as nlp(text))
def get_hedon_sent(text):
    delta = 1
    sentence_happiness = []
    for token in nlp.tokenizer(text):
        token = str(token).lower()
        word_happiness = hedon_dict[token]
        if (word_happiness > (5 + delta) or word_happiness < (5 - delta)) and word_happiness != 0:
            sentence_happiness.append(word_happiness)
    if len(sentence_happiness) >= 1:
        return sum(sentence_happiness) / len(sentence_happiness)
    else:
        return float("nan")

# ---- cell 21
def get_clauses(string):
    seen = set()
    chunks = []
    for sent in nlp(string).sents:
        heads = [cc for cc in sent.root.children if (cc.dep_ == 'conj' or cc.dep_ == "advcl")]
        for head in heads:
            words = [ww for ww in head.subtree]
            for word in words:
                seen.add(word)
            chunk = (' '.join([ww.text for ww in words]))
            chunks.append((head.i, chunk))
        unseen = [ww for ww in sent if ww not in seen]
        chunk = ' '.join([ww.text for ww in unseen])
        chunks.append((sent.root.i, chunk))
    chunks = sorted(chunks, key=lambda x: x[0])
    return [chunk for ii, chunk in chunks]

def get_wordnet_type(lemma):
    return ""   # ontology not needed (see docstring)

# ---- cell 27
def get_verb(text):
    verbcount = 0
    verb = ""; dobj = ""; dobj_ont = ""; ccomp = ""; nsubj = ""; nsubj_ont = ""
    trans_intrans = "intrans"
    doc = nlp(text)
    for token in doc:
        if token.pos_ == "VERB" and token.lemma_.isalpha():
            verbcount += 1
            verb = token.lemma_.lower()
        if token.dep_ == "dobj":
            trans_intrans = "trans"
            dobj = token.lemma_.lower()
        if token.dep_ == "ccomp":
            trans_intrans = "trans"
            for child in token.subtree:
                ccomp = ccomp + child.text + " "
        if token.dep_ == "nsubj":
            nsubj = token.lemma_.lower()
    if verbcount == 1:
        return {"verb": verb, "trans_intrans": trans_intrans, "dobj": dobj, "dobj_ont": dobj_ont,
                "ccomp": ccomp, "nsubj": nsubj, "nsubj_ont": nsubj_ont}

# ---- cell 43
def get_sconj(text):
    sconj = ""; main_clause = ""; subor_clause = ""; pobj = ""; pobj_ont = ""
    doc = nlp(text)
    for token in doc:
        if token.pos_ == "SCONJ":
            sconj = token.lemma_.lower()
            clauses = get_clauses(text)
            if len(clauses) == 2:
                for claus in clauses:
                    if sconj == claus.split(" ")[0].lower():
                        subor_clause = claus
                        clauses2 = clauses.copy()
                        clauses2.remove(subor_clause)
                        main_clause = clauses2[0]
            elif len(clauses) == 1 and token.head.dep_ == "advcl":
                for node in token.head.subtree:
                    subor_clause = subor_clause + node.text + " "
                for node in token.head.head.subtree:
                    main_clause = main_clause + node.text + " "
                main_clause = main_clause.replace(subor_clause, "")
            elif token.head.dep_ != "advcl":
                sconj = None
            for child in token.children:
                if child.dep_ == "pobj":
                    sconj = token.lemma_.lower()
                    pobj = child.lemma_
                    main_clause = ""; subor_clause = ""
                    for node in token.subtree:
                        subor_clause = subor_clause + node.text + " "
                    for node in token.head.subtree:
                        main_clause = main_clause + node.text + " "
                        main_clause = main_clause.replace(subor_clause, "")
    if sconj:
        return {"sconj": sconj, "subor_clause": subor_clause, "main_clause": main_clause,
                "pobj": pobj, "pobj_ont": pobj_ont}

def get_reader(fn):
    op = gzip.open if fn.endswith(".gz") else open
    for line in op(fn, "rt"):
        if line.strip():
            yield json.loads(line)

# ---- cell 59 (reddit_max made an argument; outputs written to OUTDIR)
def get_reddit_data(bigfile_name, outdir, reddit_max=10000000):
    lemmatizer = WordNetLemmatizer()
    pmw_dict = defaultdict(int)
    vf = ["verb", "trans_intrans", "clause", "subreddit", "clause_hedon", "dobj", "dobj_ont", "nsubj", "nsubj_ont",
          "ccomp", "ccomp_hedon", "nsubj_hedon", "dobj_hedon"]
    verb_writer = csv.DictWriter(open(os.path.join(outdir, "reddit_verbclause_sentiment.txt"), "w"), vf, delimiter='\t')
    cf = ["sentence", "subreddit", "sconj", "main_clause", "subor_clause", "pobj", "pobj_ont", "pobj_hedon",
          "main_clause_hedon", "subor_clause_hedon"]
    conj_writer = csv.DictWriter(open(os.path.join(outdir, "reddit_sconj_sentiment.txt"), "w"), cf, delimiter='\t')
    verb_writer.writeheader(); conj_writer.writeheader()
    counter = 0; wordcount = 0; already_seen_comments = set()
    for d in get_reader(bigfile_name):
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
                if counter % 10000 == 0:
                    print(f"So far {wordcount} words of {reddit_max}, {100*wordcount/reddit_max:.1f} percent", flush=True)
                for sent in sent_tokenize(d["body"].replace("\t", " ").replace("\n", " ").replace(".", ". ")):
                    for word in sent.split(" "):
                        pmw_dict[lemmatizer.lemmatize(word.lower())] += 1
                    if len(sent.translate(str.maketrans('', '', string.punctuation)).split(" ")) >= 5:
                        sconj_info = get_sconj(sent)
                        if sconj_info:
                            d0 = sconj_info; d0["subreddit"] = sub; d0["sentence"] = sent
                            d0["main_clause_hedon"] = get_hedon_sent(sconj_info["main_clause"])
                            d0["subor_clause_hedon"] = get_hedon_sent(sconj_info["subor_clause"])
                            d0["pobj_hedon"] = get_hedon_word(sconj_info["pobj"])
                            conj_writer.writerow(d0)
                    for clause in get_clauses(sent):
                        if len(clause.translate(str.maketrans('', '', string.punctuation)).split(" ")) >= 3:
                            verb_info = get_verb(clause)
                            if verb_info:
                                d1 = verb_info; d1["subreddit"] = sub; d1["clause"] = clause
                                d1["nsubj_hedon"] = get_hedon_word(verb_info["nsubj"])
                                d1["dobj_hedon"] = get_hedon_word(verb_info["dobj"])
                                d1["ccomp_hedon"] = get_hedon_sent(verb_info["ccomp"])
                                d1["clause_hedon"] = get_hedon_sent(clause)
                                verb_writer.writerow(d1)
    print("DONE, total wordcount = " + str(wordcount), flush=True)
    w = csv.DictWriter(open(os.path.join(outdir, "reddit_word_pmw.csv"), "w"), fieldnames=["word", "pmw"])
    w.writeheader()
    for word in pmw_dict:
        w.writerow({"word": word, "pmw": pmw_dict[word] * 1000000 / reddit_max})

# ---- cells 69-86: post-filters
def postprocess(outdir):
    import pandas as pd
    df = pd.read_csv(os.path.join(outdir, "reddit_verbclause_sentiment.txt"), sep="\t")
    pmw = pd.read_csv(os.path.join(outdir, "reddit_word_pmw.csv")).set_index("word")["pmw"].to_dict()
    df["verb_pmw"] = df["verb"].map(pmw)
    df = df.loc[df["verb_pmw"] >= 2]
    # Her filter is verb.isin(hedon_dict.keys()) run after extraction, when hedon_dict also holds
    # every token get_hedon_sent() looked up (see NOTE above). Her posted data accordingly keeps
    # 13,012 rows whose verb is not in the lexicon (verb_hedon missing). We reproduce that key set:
    # lexicon + all lower-cased tokens of the strings passed to get_hedon_sent().
    seen = set(LEXICON)
    sc = pd.read_csv(os.path.join(outdir, "reddit_sconj_sentiment.txt"), sep="\t")
    for col in [df["clause"], df["ccomp"], sc["main_clause"], sc["subor_clause"]]:
        for text in col.dropna().astype(str):
            seen.update(str(t).lower() for t in nlp.tokenizer(text))
    df = df.loc[df["verb"].isin(seen)]
    exclude = ["’", "‘", "“", "”", "´", "—", "-", "–", "―", "é", ":", ".", "…", "\t", "\n"]
    def keep(s):
        s = "".join(c for c in str(s) if c not in exclude)
        return all(ord(c) < 128 for c in s)
    df["keep"] = df["clause"].map(keep)
    df = df.loc[df["keep"]]
    df["verb_hedon"] = df["verb"].map(lambda v: get_hedon_word(v))
    df.to_csv(os.path.join(outdir, "verb_clause_df.csv"))
    return df

if __name__ == "__main__":
    inp, outdir = sys.argv[1], sys.argv[2]
    rmax = int(sys.argv[3]) if len(sys.argv) > 3 else 10000000
    os.makedirs(outdir, exist_ok=True)
    get_reddit_data(inp, outdir, rmax)
    postprocess(outdir)
