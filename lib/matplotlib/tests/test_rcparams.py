import os

import matplotlib as mpl
mpl.rc('text', usetex=False)
mpl.rc('lines', linewidth=22)

fname = os.path.join(os.path.dirname(__file__), 'test_rcparams.rc')
  
def test_rcparams():

    usetex = mpl.rcParams['text.usetex']
    linewidth = mpl.rcParams['lines.linewidth']

    # test context given dictionary
    with mpl.rc_context(rc={'text.usetex': not usetex}):
        assert mpl.rcParams['text.usetex'] == (not usetex)
    assert mpl.rcParams['text.usetex'] == usetex

    # test context given filename (mpl.rc sets linewdith to 33)
    with mpl.rc_context(fname=fname):
        assert mpl.rcParams['lines.linewidth'] == 33
    assert mpl.rcParams['lines.linewidth'] == linewidth

    # test context given filename and dictionary
    with mpl.rc_context(fname=fname, rc={'lines.linewidth': 44}):
        assert mpl.rcParams['lines.linewidth'] == 44
    assert mpl.rcParams['lines.linewidth'] == linewidth

    # test rc_file
    try:
        mpl.rc_file(fname)
        assert mpl.rcParams['lines.linewidth'] == 33
    finally:
        mpl.rcParams['lines.linewidth'] = linewidth
    

if __name__ == '__main__':
    test_rcparams()
