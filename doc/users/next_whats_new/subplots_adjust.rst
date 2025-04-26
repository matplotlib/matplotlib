Resetting the subplot parameters for figure.clear()
---------------------------------------------------

When calling `.Figure.clear()` the settings for `.gridspec.SubplotParams` are restored to the default values.

`~.SubplotParams.to_dict` is a new method to get the subplot parameters as a dict,
and `~.SubplotParams.reset` resets the parameters to the defaults.
