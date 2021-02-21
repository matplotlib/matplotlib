A callback registry has been added to Normalize objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`.colors.Normalize` objects now have a callback registry, ``callbacks``,
that can be connected to by other objects to be notified when the norm is
updated. The callback emits the key ``changed`` when the norm is modified.
`.cm.ScalarMappable` is now a listener and will register a change
when the norm's vmin, vmax or other attributes are changed.
