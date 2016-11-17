Two new Formatters added to `matplotlib.ticker`
-----------------------------------------------

Two new formatters have been added for displaying some specialized
tick labels:

  - :class:`matplotlib.ticker.PercentFormatter`
  - :class:`matplotlib.ticker.TransformFormatter`


:class:`matplotlib.ticker.PercentFormatter`
```````````````````````````````````````````

This new formatter has some nice features like being able to convert
from arbitrary data scales to percents, a customizable percent symbol
and either automatic or manual control over the decimal points.


:class:`matplotlib.ticker.TransformFormatter`
```````````````````````````````````````````````

A more generic version of :class:`matplotlib.ticker.FuncFormatter` that
allows the tick values to be transformed before being passed to an
underlying formatter. The transformation can yield results of arbitrary
type, so for example, using `int` as the transformation will allow
:class:`matplotlib.ticker.StrMethodFormatter` to use integer format
strings. If the underlying formatter is an instance of
:class:`matplotlib.ticker.Formatter`, it will be configured correctly
through this class.
