from matplotlib import rcParams
from matplotlib import pyplot as plt
from matplotlib.testing.decorators import cleanup

@cleanup
def test_fig_close():
    rcParams['backend'] = 'qt4agg'

    fig = plt.figure()
    fig.close()
