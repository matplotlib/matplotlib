Figure.frameon is now a direct proxy for the Figure patch visibility state
``````````````````````````````````````````````````````````````````````````

Accessing ``Figure.frameon`` (including via ``get_frameon`` and ``set_frameon``
now directly forwards to the visibility of the underlying Rectangle artist
(``Figure.patch.get_frameon``, ``Figure.patch.set_frameon``).
