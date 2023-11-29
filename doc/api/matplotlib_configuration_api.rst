**************
``matplotlib``
**************

.. py:currentmodule:: matplotlib

.. automodule:: matplotlib
   :no-members:
   :no-undoc-members:
   :noindex:

Backend management
==================

.. autofunction:: use

.. autofunction:: get_backend

.. autofunction:: interactive

.. autofunction:: is_interactive

Default values and styling
==========================

.. py:data:: rcParams

   An instance of `RcParams` for handling default Matplotlib values.

.. autoclass:: RcParams
   :no-members:

   .. automethod:: find_all
   .. automethod:: copy

.. autofunction:: rc_context

.. autofunction:: rc

.. autofunction:: rcdefaults

.. autofunction:: rc_file_defaults

.. autofunction:: rc_file

.. autofunction:: rc_params

.. autofunction:: rc_params_from_file

.. autofunction:: get_configdir

.. autofunction:: matplotlib_fname

.. autofunction:: get_data_path

Logging
=======

.. autofunction:: set_loglevel

Colormaps and color sequences
=============================

.. autodata:: colormaps
   :no-value:

.. autodata:: color_sequences
   :no-value:

Miscellaneous
=============

.. autoclass:: MatplotlibDeprecationWarning

.. autofunction:: get_cachedir
