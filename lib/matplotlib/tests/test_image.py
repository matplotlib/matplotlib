import numpy as np

from matplotlib.testing.decorators import image_comparison, knownfailureif
import matplotlib.pyplot as plt
from nose.tools import assert_raises

import cStringIO
import os

@image_comparison(baseline_images=['image_interps'])
def test_image_interps():
    'make the basic nearest, bilinear and bicubic interps'
    X = np.arange(100)
    X = X.reshape(5, 20)

    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax1.imshow(X, interpolation='nearest')
    ax1.set_title('three interpolations')
    ax1.set_ylabel('nearest')

    ax2 = fig.add_subplot(312)
    ax2.imshow(X, interpolation='bilinear')
    ax2.set_ylabel('bilinear')

    ax3 = fig.add_subplot(313)
    ax3.imshow(X, interpolation='bicubic')
    ax3.set_ylabel('bicubic')

    fig.savefig('image_interps')

def test_image_python_io():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([1,2,3])
    buffer = cStringIO.StringIO()
    fig.savefig(buffer)
    buffer.seek(0)
    plt.imread(buffer)

# def test_image_unicode_io():
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.plot([1,2,3])
#     fname = u"\u0a3a\u0a3a.png"
#     fig.savefig(fname)
#     plt.imread(fname)
#     os.remove(fname)

if __name__=='__main__':
    import nose
    nose.runmodule(argv=['-s','--with-doctest'], exit=False)
