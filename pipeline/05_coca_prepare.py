"""Step 5 (COCA): convert the COCA concordance download to the format step 6 reads.

COCA contexts mark paragraph breaks with '#' and blank out 10 words every 200 words with '@ @ @ ...' (copyright).
Both are turned into sentence boundaries ('</s><s>'), so the parser never joins words across a gap.

Input:  data/raw/coca/*.txt    concordance batches: one or more CSV blocks, each preceded by a line '### FILE <name>'
                               (see docs/concordance_format.md)
Output: data/work/coca_prepared.txt
Usage:  python pipeline/05_coca_prepare.py
"""
import csv, glob, io, re
from causesent import config

def fix(s):
    s = re.sub(r"(?:\s*@){2,}\s*", " </s><s> ", s)        # copyright gap
    s = re.sub(r"(?:^|\s)#(?=\s|$)", " </s><s> ", s)      # paragraph break
    s = re.sub(r"<p>", " </s><s> ", s)
    return re.sub(r"\s+", " ", s).strip()

config.ensure_dirs()
out = config.WORK / "coca_prepared.txt"
n = 0
with open(out, "w", encoding="utf-8") as fo:
    for src in sorted(glob.glob(str(config.RAW / "coca" / "*.txt"))):
        for part in open(src, encoding="utf-8-sig").read().split("### FILE ")[1:]:
            name, body = part.split("\n", 1)
            rows = list(csv.reader(io.StringIO(body)))
            start = next(i for i, r in enumerate(rows) if r[:1] == ["Reference"]) + 1
            w = csv.writer(fo, quoting=csv.QUOTE_ALL, lineterminator="\n")
            fo.write(f"### FILE {name.strip()}\n")
            for r in rows[:start]: w.writerow(r)
            for r in rows[start:]:
                if len(r) < 4: continue
                w.writerow([r[0], fix(r[1]), r[2], fix(r[3])])
            n += 1
print(f"{n} samples written to {out}")
