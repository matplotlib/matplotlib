Percent sign in pie labels auto-escaped with ``usetex=True``
------------------------------------------------------------

It is common, with `.Axes.pie`, to specify labels that include a percent sign
(``%``), which denotes a comment for LaTeX. When enabling LaTeX with
:rc:`text.usetex` or passing ``textprops={"usetex": True}``, this would cause
the percent sign to disappear.

Now, the percent sign is automatically escaped (by adding a preceding
backslash) so that it appears regardless of the ``usetex`` setting. If you have
pre-escaped the percent sign, this will be detected, and remain as is.
