``colorbar`` now has a *location* keyword argument
==================================================

The ``colorbar`` method now supports a *location* keyword argument to more
easily position the color bar. This is useful when providing your own inset
axes using the *cax* keyword argument and behaves similar to the case where
axes are not provided (where the *location* keyword is passed through).
*orientation* and *ticklocation* are no longer required as they are
determined by *location*. *ticklocation* can still be provided if the
automatic setting is not preferred. (*orientation* can also be provided but
must be compatible with the *location*.)

An example is:

.. plot::
    :include-source: true

    import matplotlib.pyplot as plt
    import numpy as np
    rng = np.random.default_rng(19680801)
    imdata = rng.random((10, 10))
    fig, ax = plt.subplots(layout='constrained')
    im = ax.imshow(imdata)
    fig.colorbar(im, cax=ax.inset_axes([0, 1.05, 1, 0.05]),
                 location='top')
