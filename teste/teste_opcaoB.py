import os, sys, inspect
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.path import Path
from matplotlib.patches import PathPatch


def force_paths_should_simplify(coll, value=True):
    # Mantido só pra compatibilidade; não é mais usado ativamente.
    try:
        paths = getattr(coll, "get_paths", None)
        if paths is not None:
            for p in coll.get_paths():
                p.should_simplify = value
    except Exception:
        pass


def fill_between_simplified(ax, x, y1, y2, *, color="tab:blue", alpha=0.5):
    """
    Versão de fill_between que constrói um Path, chama Path.cleaned(simplify=True)
    (respeitando path.simplify_threshold) e então cria um PathPatch preenchido.
    """
    x = np.asarray(x)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    # Polígono: (x, y1) -> volta em (x[::-1], y2[::-1])
    verts_lower = np.column_stack([x, y1])
    verts_upper = np.column_stack([x[::-1], y2[::-1]])

    verts = np.concatenate([verts_lower, verts_upper, verts_lower[0:1]], axis=0)

    # MOVETO no primeiro ponto, LINETO em todos os outros, CLOSEPOLY no último
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]

    path = Path(verts, codes)

    # Aqui é onde entra a simplificação "de verdade":
    # usa o mesmo algoritmo interno de simplify do Matplotlib, com o
    # limite controlado por rcParams['path.simplify_threshold'].
    path_simpl = path.cleaned(simplify=True)

    patch = PathPatch(path_simpl,
                      facecolor=color,
                      edgecolor="none",
                      alpha=alpha)

    ax.add_patch(patch)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(min(y1.min(), y2.min()), max(y1.max(), y2.max()))
    return patch


def gen_and_save(t_max: int, thresh: float, force_flag: bool):
    rng = np.random.default_rng(seed=424242)
    t = np.arange(1, t_max + 1, dtype=int)
    data = rng.normal(size=(10, t_max))

    # rcParams para habilitar simplificação
    mpl.rcParams['path.simplify'] = True
    mpl.rcParams['path.simplify_threshold'] = thresh

    # 1) min/max linhas (controle)
    plt.figure(dpi=200)
    plt.plot(t, np.min(data, axis=0), color='tab:blue', alpha=0.75)
    plt.plot(t, np.max(data, axis=0), color='tab:blue', alpha=0.75)
    svg_minmax = f"plot_minmax_t{t_max}_thr{thresh}_{'flag' if force_flag else 'noflag'}.svg"
    plt.savefig(svg_minmax)
    plt.close()

    # 2) fill_between simplificado (alvo)
    fig, ax = plt.subplots(dpi=200)
    ymin = np.min(data, axis=0)
    ymax = np.max(data, axis=0)

    # Aqui usamos o Path.cleaned(simplify=True), que respeita o threshold
    coll = fill_between_simplified(ax, t, ymin, ymax,
                                   color='tab:blue', alpha=0.5)

    # Mantido só pela estrutura original (não faz diferença real agora,
    # porque já estamos simplificando no Path.cleaned).
    if force_flag:
        force_paths_should_simplify(coll, True)

    svg_fill = f"plot_fill_t{t_max}_thr{thresh}_{'flag' if force_flag else 'noflag'}.svg"
    pdf_fill = f"plot_fill_t{t_max}_thr{thresh}_{'flag' if force_flag else 'noflag'}.pdf"
    plt.savefig(svg_fill)
    plt.savefig(pdf_fill)
    plt.close()

    size_minmax = os.path.getsize(svg_minmax)
    size_svg = os.path.getsize(svg_fill)
    size_pdf = os.path.getsize(pdf_fill)

    print(f"{sys.version.split()[0]},{mpl.__version__},{mpl.get_backend()},{t_max},{thresh},"
          f"{svg_minmax},{size_minmax},{svg_fill},{size_svg},{pdf_fill},{size_pdf},"
          f"{'forced_should_simplify' if force_flag else 'natural_should_simplify'}")


if __name__ == "__main__":
    print("py_ver,mpl_ver,backend,tmax,thresh,svg_minmax,size_minmax,svg_fill,size_fill,pdf_fill,size_pdf,mode")
    for tmax in (20_000, 50_000, 100_000):
        for thr in (0.0, 0.5, 1.0):
            # “Natural” (ainda usa simplificação, mas sem mexer no should_simplify manualmente)
            gen_and_save(tmax, thr, force_flag=False)
            # “Forced” (mantido só pra compatibilidade/rotulagem)
            gen_and_save(tmax, thr, force_flag=True)
