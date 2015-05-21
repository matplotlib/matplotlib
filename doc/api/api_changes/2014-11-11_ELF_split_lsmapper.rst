Split `matplotlib.cbook.ls_mapper` in two
`````````````````````````````````````````

The `matplotlib.cbook.ls_mapper` dictionary is split into two now to
distinguish between qualified linestyle used by backends and
unqualified ones. `ls_mapper` now maps from the short symbols
(e.g. `"--"`) to qualified names (`"solid"`). The new ls_mapper_r is
the reversed mapping.

