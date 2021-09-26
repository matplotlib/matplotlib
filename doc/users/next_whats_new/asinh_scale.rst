New axis scale ``asinh``
------------------------

The new ``asinh`` axis scale offers an alternative to ``symlog`` that
smoothly transitions between the quasi-linear and asymptotically logarithmic
regions of the scale. This is based on an arcsinh transformation that
allows plotting both positive and negative values than span many orders
of magnitude. A scale parameter ``a0`` is provided to allow the user
to tune the width of the linear region of the scale.
