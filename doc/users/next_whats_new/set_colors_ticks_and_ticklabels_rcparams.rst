The color of ticks and tick labels can be set independently using rcParams
--------------------------------------------------------------------------

Previously, :rc:`xtick.color` used to define the tick color and the label color. 
The label color can now be set independently using
:rc:`xtick.labelcolor`. It defaults to "inherit" which will take the value
from :rc:`xtick.color`. The same holds for ``ytick.[label]color``.
For instance, to set the ticks to light grey and the tick labels
to black, one can use the following code in a script::


    import matplotlib as mpl

    mpl.rcParams['xtick.labelcolor'] = 'lightgrey'
    mpl.rcParams['xtick.color'] = 'black'
    mpl.rcParams['ytick.labelcolor'] = 'lightgrey'
    mpl.rcParams['ytick.color'] = 'black'


Or by adding the following lines to the
:ref:`matplotlib rc <customizing-with-matplotlibrc-files>` file: or a
matplotlib style file:


.. code-block:: none

   xtick.labelcolor : lightgrey
   xtick.color      : black
   ytick.labelcolor : lightgrey
   ytick.color      : black
