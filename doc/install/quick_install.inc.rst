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

           uv usually installs its own versions of Python from the
           python-build-standalone project, and only recent versions of those
           Python builds (August 2025) work properly with the ``tkagg`` backend
           for displaying plots in a window. Please make sure you are using uv
           0.8.7 or newer (update with e.g. ``uv self update``) and that your
           bundled Python installs are up to date (with ``uv python upgrade
           --reinstall``).  Alternatively, you can use one of the other
           :ref:`supported GUI frameworks <optional_dependencies>`, e.g.

           .. code-block:: bash

               uv add matplotlib pyside6

    .. tab-item:: other

        :ref:`install-official`

        :ref:`install-third-party`

        :ref:`install-nightly-build`

        :ref:`install-source`
