.. _figures_and_backends:

++++++++++++++++++++
Figures and backends
++++++++++++++++++++

When looking at Matplotlib visualization, you are almost always looking at
Artists placed on a `~.Figure`.  In the example below, the figure is the
blue region and `~.Figure.add_subplot` has added an `~.axes.Axes` artist to the
`~.Figure` (see :ref:`figure_parts`).  A more complicated visualization can add
multiple Axes to the Figure, colorbars, legends, annotations, and the Axes
themselves can have multiple Artists added to them
(e.g. ``ax.plot`` or ``ax.imshow``).

.. plot::
    :include-source:

    fig = plt.figure(figsize=(4, 2), facecolor='lightskyblue',
                     layout='constrained')
    fig.suptitle('A nice Matplotlib Figure')
    ax = fig.add_subplot()
    ax.set_title('Axes', loc='left', fontstyle='oblique', fontsize='medium')


.. toctree::
    :maxdepth: 2

    Introduction to figures <figure_intro>

.. toctree::
    :maxdepth: 1

    Output backends <backends>
    Matplotlib Application Interfaces (APIs) <api_interfaces>
    Interacting with figures <interactive>
    Interactive figures and asynchronous programming <interactive_guide>
    Event handling <event_handling>
    Writing a backend -- the pyplot interface <writing_a_backend_pyplot_interface>
