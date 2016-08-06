`Axis.set_tick_params` now responds to 'rotation'
-------------------------------------------------

Bulk setting of tick label rotation is now possible via :func:`set_tick_params` using the `rotation` keyword. 

Example
```````
::

    ax.xaxis.set_tick_params(which='both', rotation=90)