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

    .. tab-item:: pixi

        .. code-block:: bash

            pixi add matplotlib

    .. tab-item:: uv

        .. code-block:: bash

            uv add matplotlib

        .. warning::

           If you install Python with ``uv`` then the ``tkagg`` backend
           will not be available because python-build-standalone (used by uv
           to distribute Python) does not contain tk bindings that are usable by
           Matplotlib (see `this issue`_ for details).  If you want Matplotlib
           to be able to display plots in a window, you should install one of
           the other :ref:`supported GUI frameworks <optional_dependencies>`,
           e.g.

           .. code-block:: bash

               uv add matplotlib pyside6

           .. _this issue: https://github.com/astral-sh/uv/issues/6893#issuecomment-2565965851

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
