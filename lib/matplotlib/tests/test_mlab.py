import numpy as np
import matplotlib.mlab as mlab
import tempfile
from nose.tools import raises

def test_colinear_pca():
    a = mlab.PCA._get_colinear()
    pca = mlab.PCA(a)

    assert(np.allclose(pca.fracs[2:], 0.))
    assert(np.allclose(pca.Y[:,2:], 0.))

def test_recarray_csv_roundtrip():
    expected = np.recarray((99,),
                          [('x',np.float),('y',np.float),('t',np.float)])
    expected['x'][:] = np.linspace(-1e9, -1, 99)
    expected['y'][:] = np.linspace(1, 1e9, 99)
    expected['t'][:] = np.linspace(0, 0.01, 99)
    fd = tempfile.TemporaryFile(suffix='csv')
    mlab.rec2csv(expected,fd)
    fd.seek(0)
    actual = mlab.csv2rec(fd)
    fd.close()
    assert np.allclose( expected['x'], actual['x'] )
    assert np.allclose( expected['y'], actual['y'] )
    assert np.allclose( expected['t'], actual['t'] )

@raises(ValueError)
def test_rec2csv_bad_shape():
    bad = np.recarray((99,4),[('x',np.float),('y',np.float)])
    fd = tempfile.TemporaryFile(suffix='csv')

    # the bad recarray should trigger a ValueError for having ndim > 1.
    mlab.rec2csv(bad,fd)

def test_prctile():
    # test odd lengths
    x=[1,2,3]
    assert mlab.prctile(x,50)==np.median(x)

    # test even lengths
    x=[1,2,3,4]
    assert mlab.prctile(x,50)==np.median(x)

    # derived from email sent by jason-sage to MPL-user on 20090914
    ob1=[1,1,2,2,1,2,4,3,2,2,2,3,4,5,6,7,8,9,7,6,4,5,5]
    p        = [0,   75, 100]
    expected = [1,  5.5,   9]

    # test vectorized
    actual = mlab.prctile(ob1,p)
    assert np.allclose( expected, actual )

    # test scalar
    for pi, expectedi in zip(p,expected):
        actuali = mlab.prctile(ob1,pi)
        assert np.allclose( expectedi, actuali )
