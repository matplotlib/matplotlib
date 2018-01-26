Ability to scale axis by a fixed order of magnitude
---------------------------------------------------

To scale an axis by a fixed order of magnitude, set the *scilimits* argument 
of ``Axes.ticklabel_format`` to the desired order of magnitude. Say to scale 
the y axis by a million (1e6), use ``ax.ticklabel_format(style='sci', scilimits=6, axis='y')`` 
[or equivalently ``scilimits=(6, 6)``].
