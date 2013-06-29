import os

import matplotlib as mpl
from matplotlib.tests import assert_str_equal

templaterc = os.path.join(os.path.dirname(__file__), 'test_rctemplate.rc')
  
def test_defaults():
    for k, v in mpl.rcsetup.defaultParams.iteritems():
        assert mpl.rcParams[k] == v[0]

def test_template():
    # the current matplotlibrc.template should validate successfully
    try:
        mpl.rc_file(templaterc)
        for k, v in templateParams.iteritems():
            assert mpl.rcParams[k] == v[0]

def test_unicode():
    for k, v in mpl.rcsetup.defaultParams.iteritems(): 
        v[1](unicode(v[0]))
        assert mpl.rcParams[k] == v[0]

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
