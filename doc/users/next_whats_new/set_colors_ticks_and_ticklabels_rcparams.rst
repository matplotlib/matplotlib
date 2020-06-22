The color of ticks and tick labels can be set independently from rcParams
-------------------------------------------------------------------------

The color of ticks and tick labels can now be set independently from the
matplotlib rc file or the rcParams using the :rc:`xtick.color` or 
:rc:`ytick.color` parameters to set the tick colors and the 
:rc:`xtick.labelcolor` or :rc:`ytick.labelcolor` parameters to set the tick 
label colors. For instance, to set the ticks to light grey and the tick labels
to black, one can use the following code in a script:

.. code-block:: default


    import matplotlib as mpl

    mpl.rcParams['xtick.labelcolor'] = 'lightgrey'
    mpl.rcParams['xtick.color'] = 'black'
    mpl.rcParams['ytick.labelcolor'] = 'lightgrey'
    mpl.rcParams['ytick.color'] = 'black'


Or by adding the following lines to the
:ref:`matplotlib rc <customizing-with-matplotlibrc-files>` file: or a
matplotlib style file:

   xtick.labelcolor : lightgrey
   xtick.color      : black
   ytick.labelcolor : lightgrey
   ytick.color      : black


The default value for the :rc:`xtick.labelcolor` or :rc:`ytick.labelcolor`
parameters is 'inherit', in which case the value from :rc:`xtick.color` or
:rc:`ytick.color` is used.
