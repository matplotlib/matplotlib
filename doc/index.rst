:orphan:

.. title:: Matplotlib documentation

.. module:: matplotlib

.. toctree::
   :maxdepth: 2
   :hidden:

   users/installing.rst


Matplotlib documentation
------------------------

Release: |release|

Matplotlib is a comprehensive library for creating static, animated,
and interactive visualizations in Python.

Installation
============

.. panels::
    :card: + install-card
    :column: col-lg-6 col-md-6 col-sm-12 col-xs-12 p-3

    Working with conda?
    ^^^^^^^^^^^^^^^^^^^

    Matplotlib is part of the `Anaconda <https://docs.continuum.io/anaconda/>`__
    distribution and can be installed with Anaconda or Miniconda:

    ++++++++++++++++++++++

    .. code-block:: bash

        conda install matplotlib

    ---

    Prefer pip?
    ^^^^^^^^^^^

    Matplotlib can be installed via pip from `PyPI <https://pypi.org/project/matplotlib>`__.

    ++++

    .. code-block:: bash

        pip install matplotlib

    ---
    :column: col-12 p-3

    In-depth instructions?
    ^^^^^^^^^^^^^^^^^^^^^^

    Installing a specific version? Installing from source? Check the advanced
    installation page.

    .. container:: custom-button

        :doc:`Installation Guide <users/installing>`




Learn
=====

- :doc:`Quick-start Guide <tutorials/introductory/usage>`
- Basic :doc:`Plot Types <plot_types/index>` and :doc:`Example Gallery <gallery/index>`
- `Introductory Tutorials <../tutorials/index.html#introductory>`_
- :doc:`External Learning Resources <resources/index>`


Reference
=========

- :doc:`API Reference <api/index>`

  - :doc:`pyplot API <api/pyplot_summary>`: top-level interface to create
    Figures (`.pyplot.figure`) and Subplots (`.pyplot.subplots`,
    `.pyplot.subplot_mosaic`)
  - :doc:`Axes API <api/axes_api>` for *most* plotting methods
  - :doc:`Figure API <api/figure_api>` for figure-level methods

How-tos
=======

- :doc:`Installation Guide <users/installing>`
- :doc:`Contributing to Matplotlib <devel/index>`
- :doc:`Matplotlib FAQ <faq/index>`

Understand how Matplotlib works
===============================

- The :ref:`users-guide-explain` section of the :doc:`Users guide <users/index>`
- Many of the :doc:`Tutorials <tutorials/index>` have explanatory material


Third-party Packages
--------------------

There are many `Third-party packages
<https://matplotlib.org/mpl-third-party/>`_ built on top of and extending
Matplotlib.


Contributing
------------


.. toctree::
   :maxdepth: 2

   devel/index.rst
