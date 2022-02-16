New axis scale ``asinh`` (experimental)
---------------------------------------

The new ``asinh`` axis scale offers an alternative to ``symlog`` that
smoothly transitions between the quasi-linear and asymptotically logarithmic
regions of the scale. This is based on an arcsinh transformation that
allows plotting both positive and negative values that span many orders
of magnitude.

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax0, ax1) = plt.subplots(1, 2, sharex=True)
    x = np.linspace(-3, 6, 100)

    ax0.plot(x, x)
    ax0.set_yscale('symlog')
    ax0.grid()
    ax0.set_title('symlog')

    ax1.plot(x, x)
    ax1.set_yscale('asinh')
    ax1.grid()
    ax1.set_title(r'$sinh^{-1}$')

    for p in (-2, 2):
        for ax in (ax0, ax1):
            c = plt.Circle((p, p), radius=0.5, fill=False,
                           color='red', alpha=0.8, lw=3)
            ax.add_patch(c)
