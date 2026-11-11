"""Shared figure style: Helvetica (TeX Gyre Heros, a metric-compatible Helvetica clone,
when real Helvetica is not installed). Imported by every figure script."""
import matplotlib
from matplotlib import font_manager as fm

_CANDIDATES = ["Helvetica", "Helvetica Neue", "TeX Gyre Heros", "Nimbus Sans", "Nimbus Sans L",
               "Liberation Sans", "Arial", "DejaVu Sans"]
_have = {f.name for f in fm.fontManager.ttflist}
FONT = next((c for c in _CANDIDATES if c in _have), "DejaVu Sans")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [FONT] + _CANDIDATES
matplotlib.rcParams["pdf.fonttype"] = 42      # embed TrueType/OpenType so the PDF text stays editable
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["axes.unicode_minus"] = True
