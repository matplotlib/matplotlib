.. title:: Matplotlib documentation

.. module:: matplotlib


##################################
Matplotlib |release| documentation
##################################


Matplotlib is a comprehensive library for creating static, animated,
and interactive visualizations.

Install
=======

.. tab-set::
    :class: sd-width-content-min

    .. tab-item:: pip

        .. code-block:: bash

            pip install matplotlib

    .. tab-item:: conda

        .. code-block:: bash

            conda install -c conda-forge matplotlib

    .. tab-item:: other

        .. rst-class:: section-toc
        .. toctree::
            :maxdepth: 2

            install/index

For more detailed instructions, see the
:doc:`installation guide <install/index>`.

Learn
=====

.. grid:: 1 1 2 2

    .. grid-item-card::
        :padding: 2
        :columns: 6

        **How to use Matplotlib?**
        ^^^
        .. toctree::
            :maxdepth: 1

            users/explain/quick_start
            User guide <users/index.rst>
            tutorials/index.rst
            users/faq.rst

    .. grid-item-card::
        :padding: 2
        :columns: 6

        **What can Matplotlib do?**
        ^^^
        .. toctree::
            :maxdepth: 1

            plot_types/index.rst
            gallery/index.rst


    .. grid-item-card::
        :padding: 2
        :columns: 12

        **Reference**
        ^^^

        .. grid:: 1 1 2 2
            :class-row: sd-align-minor-center

            .. grid-item::

                .. toctree::
                    :maxdepth: 1

                    API reference <api/index>
                    Figure methods <api/figure_api>
                    Plotting methods <api/axes_api>


            .. grid-item::

                Top-level interfaces to create:

                - figures: `.pyplot.figure`
                - subplots: `.pyplot.subplots`, `.pyplot.subplot_mosaic`

Community
=========

.. grid:: 1 1 2 2
    :class-row: sd-align-minor-center

    .. grid-item::

        .. rst-class:: section-toc
        .. toctree::
            :maxdepth: 2

            users/resources/index.rst

    .. grid-item::

        :octicon:`link-external;1em;sd-text-info` `Third-party packages <https://matplotlib.org/mpl-third-party/>`_,

        provide custom, domain specific, and experimental features, including
        styles, colors, more plot types and backends, and alternative
        interfaces.

What's new
==========

.. grid:: 1 1 2 2

    .. grid-item::

       Learn about new features and API changes.

    .. grid-item::

        .. toctree::
            :maxdepth: 1

            users/release_notes.rst


Contribute
==========

.. grid:: 1 1 2 2
    :class-row: sd-align-minor-center

    .. grid-item::

        Matplotlib is a community project maintained for and by its users. See
        :ref:`developers-guide-index` for the many ways you can help!

    .. grid-item::
        .. rst-class:: section-toc
        .. toctree::
            :maxdepth: 2

            devel/index.rst

About us
========

.. grid:: 1 1 2 2
    :class-row: sd-align-minor-center

    .. grid-item::

        Matplotlib was created by neurobiologist John Hunter to work with EEG
        data. It grew to be used and developed by many people in many
        different fields. John's goal was that Matplotlib make easy things easy
        and hard things possible.

    .. grid-item::
        .. rst-class:: section-toc
        .. toctree::
            :maxdepth: 2

            project/index.rst

.. _color-sequences:

Color Sequences in Matplotlib
=============================

Matplotlib provides a powerful system for defining custom color sequences,
 or color cycles, which are essential for creating consistent, visually appealing plots. 
 By customizing color sequences, users can control the color schemes used in their plots to improve accessibility and
  make their visualizations clearer.

What Are Color Sequences?
--------------------------

In Matplotlib, color sequences are essentially ordered lists of colors 
that can be used across multiple plots. They allow for consistency in the colors used
 for different lines, markers, and other visual elements in your plots. Color sequences can be customized and applied through the `cycler` module or in style sheets.

Defining a Custom Color Cycle
------------------------------

To define a custom color cycle, we use the `cycler` module 
from Matplotlib. Here's how to define and apply your own color sequence:

.. code-block:: python

    import matplotlib.pyplot as plt
    from cycler import cycler

    # Define a custom color cycle
    custom_cycle = cycler(color=["#E63946", "#F4A261", "#2A9D8F", "#264653"])

    # Apply the custom color cycle
    plt.rc("axes", prop_cycle=custom_cycle)

    # Create a simple plot to visualize the color sequence
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6], label="Line 1")
    ax.plot([1, 2, 3], [6, 7, 8], label="Line 2")
    ax.legend()
    plt.show()

Using Color Sequences in Style Sheets
--------------------------------------

You can also apply custom color sequences by defining them in a
 Matplotlib style sheet (`.mplstyle` file). This is especially useful if you want to 
 keep your color settings consistent across different projects.

Here's how you can define a custom color cycle in a style sheet:

.. code-block:: plaintext

    axes.prop_cycle: cycler(color=["#E63946", "#F4A261", "#2A9D8F", "#264653"])

Benefits of Using Custom Color Sequences
-----------------------------------------

Custom color sequences are beneficial for several reasons:

- **Consistency**: They ensure that your plots have a uniform color 
scheme across multiple figures.
- **Accessibility**: By using colorblind-friendly colors, you can make your
 visualizations more accessible to a wider audience.
- **Aesthetics**: Custom color sequences allow you to create more visually appealing
 plots that match your preferred style or brand.

Conclusion
----------

Color sequences are a powerful tool in Matplotlib for customizing the appearance of your plots.
 By defining your own color cycles and using them consistently, you can create more accessible and 
 aesthetically pleasing visualizations.
  Experiment with your own color sequences and see how they improve your plots.

For more information on colormaps and color customization, 
check out the [colormap tutorial](https://matplotlib.org/stable/tutorials/colors/colormap.html).