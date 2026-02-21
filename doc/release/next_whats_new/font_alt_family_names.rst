Fonts addressable by all their SFNT family names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fonts can now be selected by any of the family names they advertise in
the OpenType name table, not just the one FreeType reports as the primary
family name.

Some fonts store different family names on different platforms or in
different name-table entries.  For example, Ubuntu Light stores
``"Ubuntu"`` in the Macintosh-platform Name ID 1 slot (which FreeType
uses as the primary name) and ``"Ubuntu Light"`` in the Microsoft-platform
Name ID 1 slot.  Previously only the FreeType-derived name was registered,
requiring an obscure weight-based workaround::

    # Previously required
    matplotlib.rcParams['font.family'] = 'Ubuntu'
    matplotlib.rcParams['font.weight'] = 300

All name-table entries that describe a family — Name ID 1 on both
platforms, the Typographic Family (Name ID 16), and the WWS Family
(Name ID 21) — are now registered as separate entries in the
`~matplotlib.font_manager.FontManager`, so any of those names can be
used directly::

    matplotlib.rcParams['font.family'] = 'Ubuntu Light'
