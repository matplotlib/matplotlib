New drawstyles for steps - "steps-between", "steps-edges"
------------------------------------------------------------------------
They are asymmetrical such that abs(len(x) - len(y)) == 1.  Typically
to enable plotting histograms with step() and fill_between().

  .. plot::

    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(0,7,1)
    y = np.array([2,3,4,5,4,3])

    fig, ax = plt.subplots(constrained_layout=True)

    ax.plot(x, y + 2, drawstyle='steps-between')
    ax.plot(x, y, drawstyle='steps-edges')

    plt.show()

See :doc:`/gallery/lines_bars_and_markers/step_demo` and
:doc:`/gallery/lines_bars_and_markers/filled_step` for examples.
