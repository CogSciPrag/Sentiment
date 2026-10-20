"""Step 1 (Reddit): extract all r/AskReddit comments from the January 2018 Reddit comment dump, in the dump's order.

Input:  data/raw/reddit/RC_2018-01.zst   (Pushshift / Arctic Shift monthly dump, e.g. via Academic Torrents)
Output: data/work/reddit/askreddit_2018-01.jsonl.gz   (one JSON comment per line)

Needs the 'zstandard' package (pip install zstandard). Usage: python pipeline/01_reddit_extract_askreddit.py
"""
import gzip, os, time
import zstandard
from causesent import config

SRC = config.RAW / "reddit" / "RC_2018-01.zst"
OUT = config.WORK / "reddit" / "askreddit_2018-01.jsonl.gz"
NEEDLE = b'"subreddit":"AskReddit"'        # fast pre-filter; Glass's own subreddit check is applied again in step 2

OUT.parent.mkdir(parents=True, exist_ok=True)
t0 = time.time(); n_in = n_out = 0
with open(SRC, "rb") as fh, gzip.open(str(OUT) + ".tmp", "wb", compresslevel=1) as out:
    reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(fh)
    buf = b""
    while True:
        chunk = reader.read(2**24)
        if not chunk:
            break
        buf += chunk
        lines = buf.split(b"\n")
        buf = lines.pop()              # the last piece may be an incomplete line
        for line in lines:
            n_in += 1
            if NEEDLE in line:
                out.write(line + b"\n"); n_out += 1
        if n_in // 5_000_000 != (n_in - len(lines)) // 5_000_000:
            print(f"{n_in/1e6:.0f}M comments read, {n_out:,} AskReddit kept, {time.time()-t0:.0f}s", flush=True)
    if buf and NEEDLE in buf:
        out.write(buf + b"\n"); n_out += 1
os.replace(str(OUT) + ".tmp", OUT)
print(f"done: {n_in:,} comments read, {n_out:,} AskReddit comments written to {OUT} in {time.time()-t0:.0f}s")
