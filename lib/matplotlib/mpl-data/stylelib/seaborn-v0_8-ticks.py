import ast
from pathlib import Path


__mpl_style__ = {
    **ast.literal_eval(Path(__file__).with_name("_seaborn-v0_8-common.py").read_text()),
    "axes.grid": False,
    "axes.facecolor": "white",
    "axes.edgecolor": ".15",
    "axes.linewidth": 1.25,
    "grid.color": ".8",
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
}
