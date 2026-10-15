Option to use the even-odd fill rule for patches
------------------------------------------------

By default, patches such as `~.patches.Polygon` are filled according to the
`non-zero winding fill rule <https://en.wikipedia.org/wiki/Nonzero-rule>`__.
There is now the option to instead use the
`even-odd fill rule <https://en.wikipedia.org/wiki/Even%E2%80%93odd_rule>`__,
which is specified by setting the patch's ``fill_rule`` property to "evenodd".
See :doc:`/gallery/shapes_and_collections/fill_rule_demo` for more details and
an illustration of the difference.
