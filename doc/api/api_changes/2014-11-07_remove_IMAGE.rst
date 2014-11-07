Removed `Image` from main namespace
```````````````````````````````````

`Image` was imported from PIL/pillow to test if PIL is available, but there is no
reason to keep `Image` in the namespace once the availability has been determined.
