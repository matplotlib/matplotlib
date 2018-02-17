Deprecations
````````````
The `~.FigureCanvasQT.keyAutoRepeat` property is deprecated.  Directly check
``event.guiEvent.isAutoRepeat()`` in the event handler to decide whether to
handle autorepeated key presses.
