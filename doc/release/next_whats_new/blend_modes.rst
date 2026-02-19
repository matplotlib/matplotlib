Blending and compositing artists
--------------------------------

There are now alternative options for blending and compositing artists on top
of previously drawn artists, instead of the normal alpha blending.  The
behavior is controlled by the artist's ``blend_mode`` property.  See
:ref:`blend-modes` for a gallery and for a table of supporting backends.

Furthermore, there is support for blend groups, also known as transparency
groups, which can be isolated, knockout, or both.  For example, isolated blend
groups allow multiple artists to be rendered together in a separate buffer,
which is subsequently blended into the primary buffer.  See
:ref:`blend-groups` for more details and for a table of supporting backends.
