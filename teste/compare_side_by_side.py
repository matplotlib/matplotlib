from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "comparisons"
OUT_DIR.mkdir(exist_ok=True)

CASES = [
    # min/max lines
    ("plot_minmax_100000_thr0.0.png",
     "plot_minmax_100000_thr1.0.png",
     "compare_minmax_thr0_vs_thr1.png"),

    # fill_between
    ("plot_fill_100000_thr0.0.png",
     "plot_fill_100000_thr1.0.png",
     "compare_fill_thr0_vs_thr1.png"),
]


def load_rgba(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGBA")
    return np.asarray(img, dtype=float) / 255.0


def make_triplet(orig: Path, new: Path, out: Path) -> None:
    if not orig.exists():
        print(f"[SKIP] original não existe: {orig}")
        return
    if not new.exists():
        print(f"[SKIP] novo não existe: {new}")
        return

    arr_orig = load_rgba(orig)
    arr_new = load_rgba(new)

    # if different sizes: resize new one to og size.
    if arr_orig.shape != arr_new.shape:
        from PIL import Image
        img_new = Image.fromarray((arr_new * 255).astype("uint8"))
        img_new = img_new.resize(
            (arr_orig.shape[1], arr_orig.shape[0]),
            Image.LANCZOS
        )
        arr_new = np.asarray(img_new, dtype=float) / 255.0

    # avg diff by pixel (NO alpha ch)
    diff = np.abs(arr_orig[..., :3] - arr_new[..., :3]).mean(axis=2)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    axes[0].imshow(arr_orig)
    axes[0].set_title("original")
    axes[0].axis("off")

    axes[1].imshow(arr_new)
    axes[1].set_title("patch")
    axes[1].axis("off")

    im = axes[2].imshow(diff, cmap="magma")
    axes[2].set_title("abs diff")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[OK] salvo: {out}")


def main():
    for orig_name, new_name, out_name in CASES:
        orig = BASE / orig_name
        new = BASE / new_name
        out = OUT_DIR / out_name
        make_triplet(orig, new, out)


if __name__ == "__main__":
    main()