subplots_adjust has a new ``kwarg``: ``rc_default``
---------------------------------------------------

`.Figure.subplots_adjust` and `.pyplot.subplots_adjust` have a new ``kwarg``:
``rc_default`` that determines the default values for the subplot parameters.

The `.gridspec.SubplotParams` object has a new get method
:meth:`~.SubplotParams.get_subplot_params`

When calling `.Figure.clear()` the settings for `.gridspec.SubplotParams` are restored to the default values.

(code based on work by @fredrik-1)
