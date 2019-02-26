The cairo backend now defaults to pycairo instead of cairocffi
``````````````````````````````````````````````````````````````

This leads to faster import/runtime performance in some cases. The backend
will fall back to cairocffi in case pycairo isn't available.
