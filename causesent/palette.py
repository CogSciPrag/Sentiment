"""Red-green colour scale of the figures (changed 3 Nov 2026, Dean): the neutral (grey) colour is the AVERAGE OF THE FOUR
CORPORA'S BASELINES (the mean of all other verbs in enTenTen21, BNC, COCA and AskReddit) for the scoring and the measure
of the figure, not the scale's midpoint 5. Red = below that average, green = above it; the scale spans +-1.5 around it.
Before 3 Nov 2026 the centre was 5 (range 3.5-6.5)."""
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from . import config

CMAP = LinearSegmentedColormap.from_list("rg", ["#b2182b", "#e8816b", "#dcdad4", "#7fbf7b", "#1b7837"])
_P = None
def centre(sc="raw", measure="NV"):
    """Mean of the four corpora's verb baselines (combined forms) for this scoring and measure."""
    global _P
    if _P is None: _P = pd.read_csv(config.RESULTS / "pvalues_all.tsv", sep="\t")
    d = _P[(_P.scoring == sc) & (_P.measure == measure) & (_P.polarity == "combined") & (_P.baseline_type == "verbs")]
    return float(d.groupby("corpus").baseline.first().mean())
def colour_fn(c, half=1.5):
    norm = TwoSlopeNorm(vmin=c - half, vcenter=c, vmax=c + half)
    return lambda v: CMAP(norm(min(max(v, c - half), c + half)))

# ---- 3 Nov 2026 (later): a grey BAND instead of a single grey value. Every value between the lowest and the highest of the
# four corpora's baselines is grey; below the band the colour fades from grey to red (full red at 1.5 below the band),
# above it from grey to green.
def bounds(sc="raw", measure="NV"):
    """(lowest, highest) of the four corpora's verb baselines for this scoring and measure."""
    global _P
    if _P is None: _P = pd.read_csv(config.RESULTS / "pvalues_all.tsv", sep="\t")
    d = _P[(_P.scoring == sc) & (_P.measure == measure) & (_P.polarity == "combined") & (_P.baseline_type == "verbs")]
    b = d.groupby("corpus").baseline.first()
    return float(b.min()), float(b.max())
def band_colour_fn(lo, hi, half=1.5):
    def f(v):
        if v <= lo: return CMAP(0.5 * max(0.0, 1 - (lo - v) / half))
        if v >= hi: return CMAP(0.5 + 0.5 * min(1.0, (v - hi) / half))
        return CMAP(0.5)
    return f
