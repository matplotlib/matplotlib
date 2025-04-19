Resetting the subplot parameters for figure.clear()
---------------------------------------------------

When calling `.Figure.clear()` the settings for `.gridspec.SubplotParams` are restored to the default values.

The `.gridspec.SubplotParams` object has a new get method :meth:`~.SubplotParams.get_subplot_params` and a
method to reset the parameters to the defaults :meth:`~.SubplotParams.reset`


(contributed by @eendebakpt based on work by @fredrik-1)
