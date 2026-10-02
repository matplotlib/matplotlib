Axes3D annotations return Annotation3D
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Axes3D.annotate`` now constructs and returns an ``Annotation3D`` instance.
Previously, Axes3D inherited the standard 2D ``Axes.annotate`` method, which
returned a 2D ``Annotation`` fixed to the initial 2D projection and that did
not reproject when the 3D view changed. The new behavior keeps all 3D
validation and projection logic inside ``Annotation3D`` while preserving the
public API of ``Axes.annotate``.
