import os

import matplotlib as mpl
from matplotlib.tests import assert_str_equal

templaterc = os.path.join(os.path.dirname(__file__), 'test_rcsetup.rc')
  
def test_defaults():
    # the default values should be successfully set by this class
    deprecated = ['svg.embed_char_paths', 'savefig.extension']
    with mpl.rc_context(rc=mpl.rcsetup.defaultParams):
        for k, v in mpl.rcsetup.defaultParams.iteritems():
            if k not in deprecated :
                assert mpl.rcParams[k][0] == v[0]

def test_template():
    # the current matplotlibrc.template should validate successfully
    mpl.rc_file(templaterc)
    for k, v in templateParams.iteritems():
        assert mpl.rcParams[k] == v[0]

def test_unicode():
    # unicode formatted valid strings should validate.
    for k, v in mpl.rcsetup.defaultParams.iteritems(): 
        assert k == v[1](unicode(v[0]))
        assert mpl.rcParams[k] == v[0]

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
