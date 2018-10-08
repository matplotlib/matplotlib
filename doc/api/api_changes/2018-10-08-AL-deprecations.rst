Passing a Line2D's drawstyle together with the linestyle is deprecated
``````````````````````````````````````````````````````````````````````

Instead of ``plt.plot(..., linestyle="steps--")``, use ``plt.plot(...,
linestyle="--", drawstyle="steps")``. ``ds`` is now an alias for ``drawstyle``.
