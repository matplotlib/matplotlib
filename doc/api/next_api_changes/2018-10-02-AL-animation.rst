Exception on failing animations changed
```````````````````````````````````````

Previously, subprocess failures in the animation framework would raise either
in a `RuntimeError` or a `ValueError` depending on when the error occurred.
They now raise a `subprocess.CalledProcessError` with attributes set as
documented by the exception class.
