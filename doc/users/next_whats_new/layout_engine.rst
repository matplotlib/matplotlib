New ``layout_engine`` module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Matplotlib ships with ``tight_layout`` and ``constrained_layout`` layout
engines.  A new ``layout_engine`` module is provided to allow downstream
libraries to write their own layout engines and `~.figure.Figure` objects can
now take a `.LayoutEngine` subclass as an argument to the *layout* parameter.
