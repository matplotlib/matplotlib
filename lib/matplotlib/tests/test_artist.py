from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six

import io

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.transforms as mtrans
import matplotlib.collections as mcollections
from matplotlib.testing.decorators import image_comparison, cleanup


@cleanup
def test_patch_transform_of_none():
    # tests the behaviour of patches added to an Axes with various transform
    # specifications

    ax = plt.axes()
    ax.set_xlim([1, 3])
    ax.set_ylim([1, 3])

    # Draw an ellipse over data coord (2,2) by specifying device coords.
    xy_data = (2, 2)
    xy_pix = ax.transData.transform_point(xy_data)

    # Not providing a transform of None puts the ellipse in data coordinates .
    e = mpatches.Ellipse(xy_data, width=1, height=1, fc='yellow', alpha=0.5)
    ax.add_patch(e)
    assert e._transform == ax.transData

    # Providing a transform of None puts the ellipse in device coordinates.
    e = mpatches.Ellipse(xy_pix, width=120, height=120, fc='coral',
                         transform=None, alpha=0.5)
    assert e.is_transform_set() is True
    ax.add_patch(e)
    assert isinstance(e._transform, mtrans.IdentityTransform)

    # Providing an IdentityTransform puts the ellipse in device coordinates.
    e = mpatches.Ellipse(xy_pix, width=100, height=100,
                         transform=mtrans.IdentityTransform(), alpha=0.5)
    ax.add_patch(e)
    assert isinstance(e._transform, mtrans.IdentityTransform)

    # Not providing a transform, and then subsequently "get_transform" should
    # not mean that "is_transform_set".
    e = mpatches.Ellipse(xy_pix, width=120, height=120, fc='coral',
                         alpha=0.5)
    intermediate_transform = e.get_transform()
    assert e.is_transform_set() is False
    ax.add_patch(e)
    assert e.get_transform() != intermediate_transform
    assert e.is_transform_set() is True
    assert e._transform == ax.transData


@cleanup
def test_collection_transform_of_none():
    # tests the behaviour of collections added to an Axes with various
    # transform specifications

    ax = plt.axes()
    ax.set_xlim([1, 3])
    ax.set_ylim([1, 3])

    #draw an ellipse over data coord (2,2) by specifying device coords
    xy_data = (2, 2)
    xy_pix = ax.transData.transform_point(xy_data)

    # not providing a transform of None puts the ellipse in data coordinates
    e = mpatches.Ellipse(xy_data, width=1, height=1)
    c = mcollections.PatchCollection([e], facecolor='yellow', alpha=0.5)
    ax.add_collection(c)
    # the collection should be in data coordinates
    assert c.get_offset_transform() + c.get_transform() == ax.transData

    # providing a transform of None puts the ellipse in device coordinates
    e = mpatches.Ellipse(xy_pix, width=120, height=120)
    c = mcollections.PatchCollection([e], facecolor='coral',
                                     alpha=0.5)
    c.set_transform(None)
    ax.add_collection(c)
    assert isinstance(c.get_transform(), mtrans.IdentityTransform)

    # providing an IdentityTransform puts the ellipse in device coordinates
    e = mpatches.Ellipse(xy_pix, width=100, height=100)
    c = mcollections.PatchCollection([e], transform=mtrans.IdentityTransform(),
                                     alpha=0.5)
    ax.add_collection(c)
    assert isinstance(c._transOffset, mtrans.IdentityTransform)


@image_comparison(baseline_images=["clip_path_clipping"], remove_text=True)
def test_clipping():
    exterior = mpath.Path.unit_rectangle().deepcopy()
    exterior.vertices *= 4
    exterior.vertices -= 2
    interior = mpath.Path.unit_circle().deepcopy()
    interior.vertices = interior.vertices[::-1]
    clip_path = mpath.Path(vertices=np.concatenate([exterior.vertices,
                                                    interior.vertices]),
                           codes=np.concatenate([exterior.codes,
                                                 interior.codes]))

    star = mpath.Path.unit_regular_star(6).deepcopy()
    star.vertices *= 2.6

    ax1 = plt.subplot(121)
    col = mcollections.PathCollection([star], lw=5, edgecolor='blue',
                                      facecolor='red', alpha=0.7, hatch='*')
    col.set_clip_path(clip_path, ax1.transData)
    ax1.add_collection(col)

    ax2 = plt.subplot(122, sharex=ax1, sharey=ax1)
    patch = mpatches.PathPatch(star, lw=5, edgecolor='blue', facecolor='red',
                               alpha=0.7, hatch='*')
    patch.set_clip_path(clip_path, ax2.transData)
    ax2.add_patch(patch)

    ax1.set_xlim([-3, 3])
    ax1.set_ylim([-3, 3])


@cleanup
def test_cull_markers():
    x = np.random.random(20000)
    y = np.random.random(20000)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y, 'k.')
    ax.set_xlim(2, 3)

    pdf = io.BytesIO()
    fig.savefig(pdf, format="pdf")
    assert len(pdf.getvalue()) < 8000

    svg = io.BytesIO()
    fig.savefig(svg, format="svg")
    assert len(svg.getvalue()) < 20000


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
