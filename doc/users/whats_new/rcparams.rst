Added ``svg.hashsalt`` key to rcParams
```````````````````````````````````````
If ``svg.hashsalt`` is ``None`` (which it is by default), the svg backend uses ``uuid4`` to generate the hash salt.
If it is not ``None``, it must be a string that is used as the hash salt instead of ``uuid4``.
This allows for deterministic SVG output.
