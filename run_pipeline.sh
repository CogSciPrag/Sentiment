#!/usr/bin/env bash
# Runs the whole analysis in order. Every step skips work that is already done, so the script can be re-run after an
# interruption. Run it from the repository root after installing (see README) and putting the downloaded samples in
# data/raw/. To run only part of it, copy the lines you need.
#
# Rule switches (off by default; see causesent/config.py): export CAUSESENT_DOUBLE_OBJECT=1 CAUSESENT_QUANTITY=1
# CAUSESENT_NO_PRONOUNS=1 before running to use the October 2026 refinements. Use a separate CAUSESENT_DATA folder for
# each set of rules, since scored lines are not recomputed when they already exist.
set -euo pipefail
N=${N_PROCESSES:-2}     # parallel spaCy processes for the Reddit steps

# --- AskReddit, January 2018
python pipeline/01_reddit_extract_askreddit.py
python pipeline/02_reddit_sentences.py reddit_sentences_month --select study
python pipeline/02_reddit_sentences.py reddit_sentences_10M --select all --max-words 10000000
python pipeline/03_reddit_score.py "$N"
python pipeline/04_reddit_baseline.py "$N"

# --- enTenTen21 and BNC (Sketch Engine)
python pipeline/06_score_concordances.py --source sketchengine
python pipeline/07_sketchengine_baseline.py

# --- COCA
python pipeline/05_coca_prepare.py
python pipeline/06_score_concordances.py --source coca
python pipeline/08_coca_baseline.py

# --- statistics and figures
python pipeline/09_statistics.py
python pipeline/10_extra_statistics.py
for sc in raw band adj war; do
  python figures/fig_effect_sentiment.py "$sc"
  python figures/fig_N_V.py "$sc"
done
python figures/fig_distributions_and_groups.py
python figures/fig_frame_by_frame.py
python figures/fig_warriner.py
python figures/fig_wordshift.py

# --- the Glass (2023) replication: see glass_replication/README.md (it re-uses the 10-million-word sentences above)
# --- the COCA spot check: python spot_check/sample_lines.py && python spot_check/build_xlsx.py
echo "done: tables in data/results/, figures in data/figures/"
