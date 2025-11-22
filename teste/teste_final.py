import os
import numpy as np
import matplotlib as mpl
from pathlib import Path
import matplotlib.pyplot as plt
import pickle

BASE_DIR = Path(__file__).resolve().parent

pkl_path = (BASE_DIR / "data.pkl").resolve()

OUT_DIR = pkl_path.parent

t: np.ndarray
t_max: int
data: np.ndarray

with open(pkl_path, "rb") as f:
    t, t_max, data = pickle.load(f)

print("Loaded from:", pkl_path)
print("Will save outputs in:", OUT_DIR)

def run_case(thresh: float):
    # Adjusts rcParams
    mpl.rcParams['path.simplify'] = True
    mpl.rcParams['path.simplify_threshold'] = thresh

    # 1) Control: lines min/max
    plt.figure(dpi=200)
    plt.plot(t, data.min(0), alpha=.75)
    plt.plot(t, data.max(0), alpha=.75)
    plt.xlim(1, t_max)

    svg_min = OUT_DIR / f"plot_minmax_thr{thresh}.svg"
    pdf_min = OUT_DIR / f"plot_minmax_thr{thresh}.pdf"
    plt.savefig(svg_min)
    plt.savefig(pdf_min)
    plt.close()

    # 2) fill_between
    plt.figure(dpi=200)
    plt.fill_between(t, data.min(0), data.max(0), alpha=.5)
    plt.xlim(1, t_max)

    svg_fill = OUT_DIR / f"plot_fill_thr{thresh}.svg"
    pdf_fill = OUT_DIR / f"plot_fill_thr{thresh}.pdf"
    plt.savefig(svg_fill)
    plt.savefig(pdf_fill)
    plt.close()

    return {
        "svg_min": svg_min,
        "pdf_min": pdf_min,
        "svg_fill": svg_fill,
        "pdf_fill": pdf_fill,
    }

# with and without simplification
files_0 = run_case(0.0)
files_1 = run_case(1.0)

def size(path: Path) -> int:
    return os.path.getsize(path)

print("=== size (bytes) ===")
for label in ["svg_min", "pdf_min", "svg_fill", "pdf_fill"]:
    s0 = size(files_0[label])
    s1 = size(files_1[label])
    print(f"{label}: thr=0.0 -> {s0},  thr=1.0 -> {s1}")

print("\n=== ratios (thr=1.0/thr=0.0) ===")
for label in ["svg_min", "pdf_min", "svg_fill", "pdf_fill"]:
    s0 = size(files_0[label])
    s1 = size(files_1[label])
    ratio = s1 / s0 if s0 else float('nan')
    print(f"{label}: ratio = {ratio:.3f}  (reduction (approx)= {(1-ratio)*100:.1f}%)")
