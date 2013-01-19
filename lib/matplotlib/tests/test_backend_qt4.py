from matplotlib import rcParams
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import cleanup
from matplotlib._pylab_helpers import Gcf
import copy


@cleanup
def test_fig_close():
    # force switch to the Qt4 backend
    plt.switch_backend('Qt4Agg')

    #save the state of Gcf.figs
    init_figs = copy.copy(Gcf.figs)

    # make a figure using pyplot interface
    fig = plt.figure()

    # simulate user clicking the close button by reaching in
    # and calling close on the underlying Qt object
    fig.canvas.manager.window.close()

    # assert that we have removed the reference to the FigureManager
    # that got added by plt.figure()
    assert(init_figs == Gcf.figs)
