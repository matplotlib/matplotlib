++++++++++++++++
Plotting methods
++++++++++++++++

.. _users-guide-plotting:

Matplotlib can create a wide variety of data visualizations, many of which are
previewed in the :ref:`plot_types` gallery.  This section provides an overview
on working with these visualizations.  More details and examples can be found
in the :ref:`gallery` gallery, and general advice about :ref:`users_guide_axes`
and :ref:`users-guide-artists` can be found in other sections of :ref:`the
user's guide <users-guide-index>`.

.. grid:: 1 1 2 2

    .. grid-item-card::
        :padding: 2

        .. toctree::
            :maxdepth: 2
            :includehidden:

            pairwise

    .. grid-item-card::
        :padding: 2

        .. plot::
            :height: 7em

            import matplotlib.pyplot as plt
            import numpy as np

            fig, ax = plt.subplots(figsize=(3, 1.7), layout="constrained")
            rng = np.random.default_rng(seed=19680801)
            ax.plot(np.arange(200), np.cumsum(rng.normal(size=200)))
            ax.set_xlabel('time')
            ax.set_ylabel('random walk')

    .. grid-item-card::
        :padding: 2


        .. toctree::
            :maxdepth: 2
            :includehidden:

            statistical

    .. grid-item-card::
        :padding: 2

        .. plot::
            :height: 7em

            import matplotlib.pyplot as plt
            import numpy as np

            fig, ax = plt.subplots(figsize=(3.5, 1.7), layout="constrained")
            rng = np.random.default_rng(seed=19680801)
            ax.hist(rng.normal(size=200), density=True)
            ax.set_xlabel('x')
            ax.set_ylabel('pdf')

    .. grid-item-card::
        :padding: 2

        .. toctree::
            :maxdepth: 2
            :includehidden:

            gridded

    .. grid-item-card::
        :padding: 2


        .. plot::
            :height: 7em

            import matplotlib.pyplot as plt
            import numpy as np


            # make data with uneven sampling in x
            x = [-3, -2, -1.6, -1.2, -.8, -.5, -.2, .1, .3, .5, .8, 1.1, 1.5, 1.9, 2.3, 3]
            X, Y = np.meshgrid(x, np.linspace(-3, 3, 128))
            Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)

            fig, ax = plt.subplots(figsize=(3.5, 1.7), layout="constrained")

            ax.pcolormesh(X, Y, Z, vmin=-0.5, vmax=1.0)
            ax.set_xlabel('x')
            ax.set_ylabel('y')

            ax.contour(X, Y, Z, vmin=-0.5, vmax=1.0, linewidths=2, cmap='RdBu_r')


    .. grid-item-card::
        :padding: 2

        .. toctree::
            :maxdepth: 2
            :includehidden:

            unstructuredgrid

    .. grid-item-card::
        :padding: 2

        .. plot::
            :height: 7em

            import matplotlib.pyplot as plt
            import numpy as np

            # make data:
            np.random.seed(1)
            x = np.random.uniform(-3, 3, 256)
            y = np.random.uniform(-3, 3, 256)
            z = (1 - x/2 + x**5 + y**3) * np.exp(-x**2 - y**2)

            # plot:
            fig, ax = plt.subplots(figsize=(3.5, 1.7), layout='constrained')

            ax.plot(x, y, 'o', markersize=2, color='grey')
            ax.tripcolor(x, y, z)
            ax.tricontour(x, y, z, cmap='RdBu_r', linewidths=2)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set(xlim=(-3, 3), ylim=(-3, 3))

            plt.show()


    .. grid-item-card::
        :padding: 2

        .. toctree::
            :maxdepth: 2
            :includehidden:

            threed


    .. grid-item-card::
        :padding: 2


        .. plot::
            :height: 7em

            import matplotlib.pyplot as plt
            import numpy as np

            # Make data
            X = np.arange(-5, 5, 0.25)
            Y = np.arange(-5, 5, 0.25)
            X, Y = np.meshgrid(X, Y)
            R = np.sqrt(X**2 + Y**2)
            Z = np.sin(R)

            # Plot the surface
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"},
                                   layout='constrained', figsize=(4*0.8, 2.7*0.8))
            ax.plot_surface(X, Y, Z, vmin=Z.min() * 2)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
