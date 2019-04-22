Errorbar plots can shift which points have error bars
-----------------------------------------------------

Previously, `plt.errorbar()` accepted a kwarg `errorevery` such that the
command `plt.errorbar(x, y, yerr, errorevery=6)` would add error bars to
datapoints `x[::6], y[::6]`.

`errorbar()` now also accepts a tuple for `errorevery` such that
`plt.errorbar(x, y, yerr, errorevery=(start, N))` adds error bars to points
`x[start::N], y[start::N]`.
