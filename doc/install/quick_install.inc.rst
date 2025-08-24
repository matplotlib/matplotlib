.. set of quick install commands for reuse across docs

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

        :ref:`install-official`

        :ref:`install-third-party`

        :ref:`install-nightly-build`

        :ref:`install-source`
