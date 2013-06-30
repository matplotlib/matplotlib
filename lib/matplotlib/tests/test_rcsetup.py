import os

import matplotlib as mpl
from matplotlib.tests import assert_str_equal

templaterc = os.path.join(os.path.dirname(__file__), 'test_rcsetup.rc')
deprecated = ['svg.embed_char_paths', 'savefig.extension']

def test_defaults():
    # the default values should be successfully set by this class
    with mpl.rc_context(rc=mpl.rcsetup.defaultParams):
        for k, v in mpl.rcsetup.defaultParams.iteritems():
            if k not in deprecated:
                assert mpl.rcParams[k][0] == v[0]


def test_template():
    # the current matplotlibrc.template should validate successfully
    with mpl.rc_context(fname=templaterc):
        for k, v in mpl.rcsetup.defaultParams.iteritems():
            if k not in deprecated:
                if mpl.rcParams[k] != v[0]:
                    print k
                    print "Expected : ", v[0]
                    print "Expected type", type(v[0])
                    print "Actual : ", mpl.rcParams[k]
                    print "Actual type : ", type(mpl.rcParams[k])
                    print "---------------"
                if isinstance(v[0], basestring):
                    assert mpl.rcParams[k] in [v[0], v[0].lower()]
                else : 
                    assert mpl.rcParams[k] == v[0]

def test_unicode():
    # unicode formatted valid strings should validate.
    for k, v in mpl.rcsetup.defaultParams.iteritems():
        assert k == v[1](unicode(v[0]))
        assert mpl.rcParams[k] == v[0]

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=['-s', '--with-doctest'], exit=False)
