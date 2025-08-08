import matplotlib.pyplot as plt
import numpy as np

def test_imshow_alpha_array():
    data = np.random.rand(10, 10)
    alpha = np.linspace(0, 1, 100).reshape(10, 10)

    fig, ax = plt.subplots()
    im = ax.imshow(data, alpha=alpha)
    plt.colorbar(im, ax=ax)
    return fig
