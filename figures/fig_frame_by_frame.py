"""Figure: each expression against the AskReddit baseline within the same frame (does the difference hold frame by frame?).
Input: data/results/extra/ (step 10). Output: data/figures/. Usage: python figures/fig_frame_by_frame.py
"""
import numpy as np, pandas as pd, matplotlib
from causesent import config
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from causesent.targets import PREVENTION, EFFECT_LEFT
EXTRA = str(config.RESULTS / "extra") + "/"; FIG = str(config.FIGURES) + "/"; config.ensure_dirs()
F = pd.read_csv(EXTRA + "frame_by_frame_vs_baseline.tsv", sep="\t"); S = pd.read_csv(EXTRA + "word_stats.tsv", sep="\t")
CM = LinearSegmentedColormap.from_list("rg", ["#b2182b", "#e8816b", "#f4f3ef", "#7fbf7b", "#1b7837"])
lab = lambda it: it + (" *" if it == "make" else "") + (" †" if it in PREVENTION else "") + (" ←" if it in EFFECT_LEFT else "")
fig, axes = plt.subplots(1, 2, figsize=(16, 14), sharey=True)
order = S[(S.corpus == "reddit") & (S.scoring == "raw")].sort_values("diff").item.tolist()
order = [o for o in order if o in set(F[F.scoring == "raw"].item)]
frames = F[F.scoring == "raw"].frame.value_counts().index.tolist()
for ax, sc in zip(axes, ["raw", "band"]):
    P = F[F.scoring == sc].pivot(index="item", columns="frame", values="diff").reindex(index=order, columns=frames)
    N = F[F.scoring == sc].pivot(index="item", columns="frame", values="n").reindex(index=order, columns=frames)
    im = ax.imshow(P.values, cmap=CM, norm=TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5), aspect="auto")
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            v = P.values[i, j]
            if not np.isnan(v): ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=6, color="#0b0b0b")
    ax.set_xticks(range(len(frames))); ax.set_xticklabels(frames, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([lab(i) for i in order], fontsize=7.8, fontstyle="italic")
    ax.set_title(f"{'A' if sc == 'raw' else 'B'}. {'Raw scores' if sc == 'raw' else 'Neutral band (4–6) removed'}", fontsize=10, loc="left")
    for s in ax.spines.values(): s.set_visible(False)
cax = fig.add_axes([0.35, 0.06, 0.3, 0.012]); fig.colorbar(im, cax=cax, orientation="horizontal", label="difference from the same frame for all other verbs")
fig.suptitle("Frame by frame: each word compared with all other verbs in the same frame (AskReddit)", fontsize=13, x=0.01, ha="left")
fig.text(0.01, 0.005, "Cells = word mean minus the mean of all other verbs in that frame (both ≥ 30 lines).\nBlank = too few lines, or the frame does not exist for ordinary verbs, "
         "e.g. V + PREP + N and effect-on-the-left frames). Rows sorted by the overall raw difference. Words with no comparable frame (e.g. lead to, result in, because of) are left out.", fontsize=7.4, color="#52514e", va="bottom")
fig.subplots_adjust(left=0.12, right=0.99, top=0.95, bottom=0.17, wspace=0.05)
fig.savefig(FIG + "fig6_frame_by_frame.png", dpi=200); fig.savefig(FIG + "fig6_frame_by_frame.pdf"); print("ok")
