Unique identifier added to `RendererBase` classes
`````````````````````````````````````````````````

Since ``id()`` is not guaranteed to be unique between objects that exist at
different times, a new private property ``_uid`` has been added to
`RendererBase` which is used along with the renderer's ``id()`` to cache
certain expensive operations.

If a custom renderer does not subclass `RendererBase` or `MixedModeRenderer`,
it is not required to implement this ``_uid`` property, but this may produce
incorrect behavior when the renderers' ``id()`` clashes.
