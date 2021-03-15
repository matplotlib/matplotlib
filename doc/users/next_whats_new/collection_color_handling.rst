Collection color specification and mapping
------------------------------------------

Reworking the handling of color mapping and the keyword arguments for facecolor
and edgecolor has resulted in three behavior changes:

1.  Color mapping can be turned off by calling ``Collection.set_array(None)``.
    Previously, this would have no effect.
2.  When a mappable array is set, with ``facecolor='none'`` and
    ``edgecolor='face'``, both the faces and the edges are left uncolored.
    Previously the edges would be color-mapped.
3.  When a mappable array is set, with ``facecolor='none'`` and
    ``edgecolor='red'``, the edges are red.  This addresses Issue #1302.
    Previously the edges would be color-mapped.
