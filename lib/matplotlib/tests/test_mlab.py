import numpy as np
import matplotlib.mlab as mlab

@staticmethod
def test_colinear_pca():
    a = mlab.PCA._get_colinear()
    pca = mlab.PCA(a)

    assert(np.allclose(pca.fracs[2:], 0.))
    assert(np.allclose(pca.Y[:,2:], 0.))

