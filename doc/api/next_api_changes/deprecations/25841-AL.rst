``Tick.set_label1`` and ``Tick.set_label2``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
... are deprecated.  Calling these methods from third-party code usually has no
effect, as the labels are overwritten at draw time by the tick formatter.
