
import ast
from pathlib import Path


__mpl_style__ = {
    **ast.literal_eval(Path(__file__).with_name("_seaborn-v0_8-common.py").read_text()),
    "axes.grid": False,
    "axes.facecolor": "white",
    "axes.edgecolor": ".15",
    "axes.linewidth": 1.25,
    "grid.color": ".8",
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,
}
