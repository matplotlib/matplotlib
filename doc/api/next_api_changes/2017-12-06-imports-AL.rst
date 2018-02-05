Removal of unused imports
`````````````````````````
Many unused imports were removed from the codebase.  As a result,
trying to import certain classes or functions from the "wrong" module
(e.g. `~.Figure` from :mod:`matplotlib.backends.backend_agg` instead of
:mod:`matplotlib.figure`) will now raise an `ImportError`.
