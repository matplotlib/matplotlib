``FixedLocator`` rejects non-finite locations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`~matplotlib.ticker.FixedLocator` now raises a `ValueError` at
construction time if its locations contain NaN or infinity.
Previously, these inputs could cause errors later during figure
rendering.
