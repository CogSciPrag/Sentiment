"""Convert the Hedonometer word list labMT-en-v2, downloaded as JSON from the hedonometer.org API, to the TSV the
pipeline reads (columns word, rank, happs, stddev).

Download (all 10,187 words in one request):
    curl -o lexicons/labMT-en-v2.json \
      "https://hedonometer.org/api/v1/words/?format=json&wordlist__title=labMT-en-v2&limit=20000"
Then, from the repository root:
    python lexicons/labmt_json_to_tsv.py lexicons/labMT-en-v2.json lexicons/labMT-en-v2.tsv
"""
import csv, json, sys

src, dst = (sys.argv[1:3] + [None, None])[:2]
src = src or "lexicons/labMT-en-v2.json"; dst = dst or "lexicons/labMT-en-v2.tsv"
j = json.load(open(src, encoding="utf-8"))
words = j["objects"]
assert len(words) == j["meta"]["total_count"], "incomplete download: raise the limit in the request"
with open(dst, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["word", "rank", "happs", "stddev"])
    for o in sorted(words, key=lambda o: o["rank"]):
        w.writerow([o["word"], o["rank"], o["happs"], o["stdDev"]])
print(f"{len(words)} words ({words[0]['wordlist']['title']}, {words[0]['wordlist']['date']}) -> {dst}")
