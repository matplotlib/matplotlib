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
   :type: RcParams

   The global configuration settings for Matplotlib.

   This a dictionary-like variable that stores the current configuration
   settings.  Many of the values control styling, but others control
   various aspects of Matplotlib's behavior.

   See :doc:`/users/explain/configuration` for a full list of config
   parameters.

   See :ref:`customizing` for usage information.

   Notes
   -----
   This object is also available as ``plt.rcParams`` via the
   `matplotlib.pyplot` module (which by convention is imported as ``plt``).


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
