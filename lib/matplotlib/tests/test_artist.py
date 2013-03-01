from __future__ import print_function

from matplotlib.testing.decorators import cleanup

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
import matplotlib.transforms as mtrans
import matplotlib.collections as mcollections


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
    

if __name__=='__main__':
    import nose
    nose.runmodule(argv=['-s','--with-doctest'], exit=False)
