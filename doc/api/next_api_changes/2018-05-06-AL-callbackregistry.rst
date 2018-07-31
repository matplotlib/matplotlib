`CallbackRegistry` now stores callbacks using stdlib's `WeakMethod`\s
`````````````````````````````````````````````````````````````````````

In particular, this implies that ``CallbackRegistry.callbacks[signal]`` is now
a mapping of callback ids to `WeakMethod`\s (i.e., they need to be first called
with no arguments to retrieve the method itself).
