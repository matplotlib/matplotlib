Allow "real" LaTeX code for ``pgf.preamble`` in matplotlib rc file
``````````````````````````````````````````````````````````````````

Previously, the rc file key ``pgf.preamble`` was parsed using commmas as
separators. This would break valid LaTeX code, such as::

\usepackage[protrusion=true, expansion=false]{microtype}

The parsing has been modified to pass the complete line to the LaTeX system,
keeping all commas.

Passing a list of strings from within a Python script still works as it used to.
