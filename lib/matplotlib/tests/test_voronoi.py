'''Test suite for Voronoi diagrams.'''

import numpy as np
from nose.tools import assert_in

import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
from matplotlib.tri import compute_voronoi_cells


# Common variables
N = 128
np.random.seed(42)
x, y = np.random.random((2, N))
c = (x ** 2 + y ** 2) ** 0.25


# Tests
def test_voronoi_cell():
    '''
    This test simply checks whether the Voronoi cells are created properly.
    '''
    x = np.array([-1.0, 0.0, 1.0, 0.0])
    y = np.array([-1.0, 1.0, -1.0, 0.0])
    cells = compute_voronoi_cells(x, y)

    assert_in((1.5, 0.5), cells[3])
    assert_in((-1.5, 0.5), cells[3])
    assert_in((0.0, -1.0), cells[3])


@image_comparison(baseline_images=['voronoi_pseudocolor_image'])
def test_voronoi_pseudocolor():
    '''
    Image comparison test of pseudocolored Voronoi diagram.
    '''
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    plt.voronoi(x, y, c)
    ax.tick_params(labelleft=False, labelbottom=False)


@image_comparison(baseline_images=['voronoi_simple_image'])
def test_voronoi_simple():
    '''
    Image comparison test of a Voronoi diagram that illustrates the relation
    between cells and the unstructured grid.
    '''
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    plt.voronoi(x, y, facecolors='white')
    plt.scatter(x, y)
    ax.tick_params(labelleft=False, labelbottom=False)


# Main routine
if __name__=='__main__':
    import nose
    nose.runmodule(argv=['-s','--with-doctest'], exit=False)

