Add ``Annulus`` patch
---------------------

`.Annulus` is a new class for drawing elliptical annuli.

.. plot::

    import matplotlib.pyplot as plt
    from matplotlib.patches import Annulus

    fig, ax = plt.subplots()
    cir = Annulus((0.5, 0.5), 0.2, 0.05, fc='g')        # circular annulus
    ell = Annulus((0.5, 0.5), (0.5, 0.3), 0.1, 45,      # elliptical
                fc='m', ec='b', alpha=0.5, hatch='xxx') 
    ax.add_patch(cir)
    ax.add_patch(ell)
    ax.set_aspect('equal')
