Allow tick formatters to be set with str or function inputs
------------------------------------------------------------------------
`~.Axis.set_major_formatter` and `~.Axis.set_minor_formatter`
now accept `str` or function inputs in addition to `~.ticker.Formatter`
instances. For a `str` a `~.ticker.StrMethodFormatter` is automatically
generated and used. For a function a `~.ticker.FuncFormatter` is automatically
generated and used.
