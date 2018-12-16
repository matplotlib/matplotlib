Deprecations
````````````

The class variable ``matplotlib.ticker.MaxNLocator.default_params`` is
deprecated and will be removed in a future version. The defaults are not
supposed to be user-configurable.

``matplotlib.ticker.MaxNLocator`` and its ``set_params`` method will issue
a warning on unknown keyword arguments instead of silently ignoring them.
Future versions will raise an error.
