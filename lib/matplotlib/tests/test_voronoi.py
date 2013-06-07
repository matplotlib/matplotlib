import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
import numpy as np

# Common variables
N = 128
np.random.seed(42)
x, y = np.random.random((2, N))
c = (x ** 2 + y ** 2) ** 0.25

# Tests
@image_comparison(baseline_images=['voronoi_pseudocolor_image'])
def test_voronoi_pseudocolor():
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    plt.voronoi(x, y, c)
    ax.tick_params(labelleft=False, labelbottom=False)

@image_comparison(baseline_images=['voronoi_simple_image'])
def test_voronoi_simple():
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    plt.voronoi(x, y, facecolors='white')
    plt.scatter(x, y)
    ax.tick_params(labelleft=False, labelbottom=False)

# Main routine
if __name__=='__main__':
    import nose
    nose.runmodule(argv=['-s','--with-doctest'], exit=False)

