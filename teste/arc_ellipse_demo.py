# teste/arc_ellipse_demo.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def make_ellipse(threshold: float, fname: str):
    mpl.rcParams["path.simplify"] = True
    mpl.rcParams["path.simplify_threshold"] = threshold

    fig, ax = plt.subplots(figsize=(4, 3), dpi=200)

    x0, y0 = 0.0, 0.0
    a, b = 3.0, 1.5
    theta = np.deg2rad(30)

    t = np.linspace(0, 2*np.pi, 361)
    x = x0 + a*np.cos(t)*np.cos(theta) - b*np.sin(t)*np.sin(theta)
    y = y0 + a*np.cos(t)*np.sin(theta) + b*np.sin(t)*np.cos(theta)

    ax.fill(x, y, facecolor="#ffffcc", edgecolor="black")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-3, 3)

    ax.set_title(f"ellipse thr={threshold}")
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    make_ellipse(0.0, "ellipse_thr0.0.png")
    make_ellipse(1.0, "ellipse_thr1.0.png")