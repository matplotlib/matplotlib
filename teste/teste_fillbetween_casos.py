import os
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "out_fill_casos"
OUT_DIR.mkdir(exist_ok=True)

print("Saving outputs to:", OUT_DIR)

# Gera dados sintéticos controlados (independe do data.pkl)
n = 2000
t = np.linspace(0, 10, n)
f1 = np.sin(t)
f2 = 0.2 * np.cos(2 * t)
f3 = np.sin(t) + 0.5  # para casos com curva deslocada
f4 = -0.5 * np.ones_like(t)  # base constante

# Alguns padrões de 'where'
where_all = np.ones_like(t, dtype=bool)
where_even = np.zeros_like(t, dtype=bool)
where_even[::2] = True
where_blocks = np.zeros_like(t, dtype=bool)
where_blocks[n//10:2*n//10] = True
where_blocks[5*n//10:7*n//10] = True

rng = np.random.default_rng(4242)
where_random = rng.random(n) > 0.3  # regiões aleatórias


def make_crossing_curves():
    # f1 e f2 se cruzam algumas vezes
    c1 = np.sin(t)
    c2 = np.cos(t) * 0.9
    return c1, c2


# Cada cenário é (nome, kwargs_para_fill_between, descrição_curva)
SCENARIOS = []

# 1) Básico: sem where/step/interpolate
SCENARIOS.append((
    "basic",
    dict(y1=f1, y2=f2, where=None, step=None, interpolate=False,
         color="C0", alpha=0.5),
    "sin vs 0.2cos"
))

# 2) with where: blocos grandes
SCENARIOS.append((
    "where_blocks",
    dict(y1=f1, y2=f4, where=where_blocks, step=None, interpolate=False,
         color="C1", alpha=0.5),
    "sin vs const -0.5, where blocks"
))

# 3) where alternado (muitos pedacinhos)
SCENARIOS.append((
    "where_even",
    dict(y1=f1, y2=np.zeros_like(t), where=where_even, step=None,
         interpolate=False, color="C2", alpha=0.5),
    "sin vs 0, where every other sample"
))

# 4) where aleatório (regiões quebradas)
SCENARIOS.append((
    "where_random",
    dict(y1=f1, y2=f2, where=where_random, step=None, interpolate=False,
         color="C3", alpha=0.5),
    "sin vs 0.2cos, random where"
))

# 5) step='pre'
SCENARIOS.append((
    "step_pre",
    dict(y1=f1, y2=f2, where=None, step="pre", interpolate=False,
         color="C4", alpha=0.5),
    "sin vs 0.2cos, step=pre"
))

# 6) step='post'
SCENARIOS.append((
    "step_post",
    dict(y1=f1, y2=f2, where=None, step="post", interpolate=False,
         color="C5", alpha=0.5),
    "sin vs 0.2cos, step=post"
))

# 7) step='mid'
SCENARIOS.append((
    "step_mid",
    dict(y1=f1, y2=f2, where=None, step="mid", interpolate=False,
         color="C6", alpha=0.5),
    "sin vs 0.2cos, step=mid"
))

# 8) interpolate=True com curvas que se cruzam
c1, c2 = make_crossing_curves()
SCENARIOS.append((
    "interpolate_cross",
    dict(y1=c1, y2=c2, where=c1 > c2, step=None, interpolate=True,
         color="C7", alpha=0.5),
    "sin vs cos, interpolate where sin>cos"
))

# 9) múltiplas regiões & cores no mesmo Axes
def draw_multi(ax):
    ax.fill_between(t, f1, f2, where=where_blocks, color="C0", alpha=0.5)
    ax.fill_between(t, f3, f4, where=where_random, color="C1", alpha=0.5)
    ax.fill_between(t, f2, f4, where=where_even,   color="C2", alpha=0.5)
    ax.set_xlim(t[0], t[-1])

MULTI_NAME = "multi_regions"


def run_case(thresh: float):
    mpl.rcParams["path.simplify"] = True
    mpl.rcParams["path.simplify_threshold"] = thresh

    out_files = {}

    # Casos 1–8: um fill_between por figura
    for name, fb_kwargs, desc in SCENARIOS:
        fig, ax = plt.subplots(dpi=200)
        ax.plot(t, fb_kwargs["y1"], lw=0.4, color="black", alpha=0.3)
        if fb_kwargs["y2"] is not None:
            y2 = fb_kwargs["y2"]
            if np.isscalar(y2):
                y2_plot = np.full_like(t, y2, dtype=float)
            else:
                y2_plot = y2
            ax.plot(t, y2_plot, lw=0.4, color="black", alpha=0.3)

        ax.fill_between(
            t,
            fb_kwargs["y1"],
            fb_kwargs["y2"],
            where=fb_kwargs["where"],
            step=fb_kwargs["step"],
            interpolate=fb_kwargs["interpolate"],
            color=fb_kwargs["color"],
            alpha=fb_kwargs["alpha"],
        )
        ax.set_title(f"{name} (thr={thresh})")
        ax.set_xlim(t[0], t[-1])

        svg_path = OUT_DIR / f"fill_{name}_thr{thresh}.svg"
        pdf_path = OUT_DIR / f"fill_{name}_thr{thresh}.pdf"
        fig.savefig(svg_path)
        fig.savefig(pdf_path)
        plt.close(fig)

        out_files.setdefault(name, {})["svg"] = svg_path
        out_files[name]["pdf"] = pdf_path

    # Caso 9: várias regiões/cores no mesmo Axes
    fig, ax = plt.subplots(dpi=200)
    draw_multi(ax)
    ax.set_title(f"{MULTI_NAME} (thr={thresh})")
    svg_path = OUT_DIR / f"fill_{MULTI_NAME}_thr{thresh}.svg"
    pdf_path = OUT_DIR / f"fill_{MULTI_NAME}_thr{thresh}.pdf"
    fig.savefig(svg_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    out_files.setdefault(MULTI_NAME, {})["svg"] = svg_path
    out_files[MULTI_NAME]["pdf"] = pdf_path

    return out_files


def size(path: Path) -> int:
    return os.path.getsize(path)


files_thr0 = run_case(0.0)
files_thr1 = run_case(1.0)

print("\n=== size (bytes) ===")
for name in sorted(files_thr0.keys()):
    s0_svg = size(files_thr0[name]["svg"])
    s1_svg = size(files_thr1[name]["svg"])
    s0_pdf = size(files_thr0[name]["pdf"])
    s1_pdf = size(files_thr1[name]["pdf"])
    print(f"{name:20s}  SVG thr0={s0_svg:8d}, thr1={s1_svg:8d} | "
          f"PDF thr0={s0_pdf:8d}, thr1={s1_pdf:8d}")

print("\n=== ratios (thr=1.0 / thr=0.0) ===")
for name in sorted(files_thr0.keys()):
    s0_svg = size(files_thr0[name]["svg"])
    s1_svg = size(files_thr1[name]["svg"])
    s0_pdf = size(files_thr0[name]["pdf"])
    s1_pdf = size(files_thr1[name]["pdf"])

    r_svg = s1_svg / s0_svg if s0_svg else float("nan")
    r_pdf = s1_pdf / s0_pdf if s0_pdf else float("nan")

    print(f"{name:20s}  SVG ratio={r_svg:6.3f} "
          f"(red ≈ {(1 - r_svg) * 100:5.1f}%)  |  "
          f"PDF ratio={r_pdf:6.3f} "
          f"(red ≈ {(1 - r_pdf) * 100:5.1f}%)")
