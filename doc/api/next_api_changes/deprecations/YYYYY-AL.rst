rcParams.copy() will return a new RcParams instance in the future
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
During the transition period, ``rcParams.copy()`` will emit a
DeprecationWarning.  Either use ``dict.copy(rcParams)`` to copy rcParams as
a plain dict, or ``copy.copy(rcParams)`` to copy rcParams as a new RcParams
instance.
