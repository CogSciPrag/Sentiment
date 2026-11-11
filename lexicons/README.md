# Lexicons

Put these files in this folder (they are not redistributed here):

- `labMT-en-v2.tsv`: the Hedonometer word list labMT-en-v2 (hedonometer.org, 2020-03-28; 10,187 words), the list
  Glass (2023) used. Download it as JSON from the hedonometer.org API and convert it with the script here:

  ```bash
  curl -o lexicons/labMT-en-v2.json \
    "https://hedonometer.org/api/v1/words/?format=json&wordlist__title=labMT-en-v2&limit=20000"
  python lexicons/labmt_json_to_tsv.py lexicons/labMT-en-v2.json lexicons/labMT-en-v2.tsv
  ```

  Tab-separated, with columns `word`, `rank`, `happs` (1-9) and `stddev`. Its scores are those of labMT 1.0
  (Dodds et al. 2011) for every shared word; it adds 17 pandemic words and drops 52 Twitter hashtags and handles.
- `labMT2english.txt` (optional): the list of the first results (10,222 words; the labMT 1.0 list), from the
  `andyreagan/sentidict` repository: <https://github.com/andyreagan/sentidict/blob/main/data/LabMT/labMT2english.txt.gz>
  (download and unzip: `gunzip labMT2english.txt.gz`). Used only with `CAUSESENT_LABMT=labMT2english.txt`.
- `warriner_2013.csv`: valence norms of Warriner, Kuperman & Brysbaert (2013), supplementary file
  `Ratings_Warriner_et_al.csv` (renamed). Only used for the robustness check with a second lexicon
  (columns `Word` and `V.Mean.Sum`).
