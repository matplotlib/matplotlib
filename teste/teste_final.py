import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

rng = np.random.default_rng(424242)
t_max = 100_000
t = np.arange(1, t_max + 1)
data = rng.normal(size=(10, t_max))

def run_case(thresh: float):
    # Ajusta rcParams para este caso
    mpl.rcParams['path.simplify'] = True
    mpl.rcParams['path.simplify_threshold'] = thresh

    # 1) Controle: linhas min/max
    plt.figure(dpi=200)
    plt.plot(t, data.min(0), alpha=.75)
    plt.plot(t, data.max(0), alpha=.75)
    plt.xlim(1, t_max)

    svg_min = f"plot_minmax_thr{thresh}.svg"
    pdf_min = f"plot_minmax_thr{thresh}.pdf"
    plt.savefig(svg_min)
    plt.savefig(pdf_min)
    plt.close()

    # 2) fill_between – alvo
    plt.figure(dpi=200)
    plt.fill_between(t, data.min(0), data.max(0), alpha=.5)
    plt.xlim(1, t_max)

    svg_fill = f"plot_fill_thr{thresh}.svg"
    pdf_fill = f"plot_fill_thr{thresh}.pdf"
    plt.savefig(svg_fill)
    plt.savefig(pdf_fill)
    plt.close()

    return {
        "svg_min": svg_min,
        "pdf_min": pdf_min,
        "svg_fill": svg_fill,
        "pdf_fill": pdf_fill,
    }

# roda sem simplificação e com simplificação máxima
files_0 = run_case(0.0)
files_1 = run_case(1.0)

def size(path): return os.path.getsize(path)

print("=== tamanhos absolutos (bytes) ===")
for label in ["svg_min", "pdf_min", "svg_fill", "pdf_fill"]:
    s0 = size(files_0[label])
    s1 = size(files_1[label])
    print(f"{label}: thr=0.0 -> {s0},  thr=1.0 -> {s1}")

print("\n=== razões (thr=1.0 / thr=0.0) ===")
for label in ["svg_min", "pdf_min", "svg_fill", "pdf_fill"]:
    s0 = size(files_0[label])
    s1 = size(files_1[label])
    ratio = s1 / s0 if s0 else float('nan')
    print(f"{label}: ratio = {ratio:.3f}  (redução ≈ {(1-ratio)*100:.1f}%)")
