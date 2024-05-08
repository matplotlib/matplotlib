Deprecations
------------

``plot_date``
^^^^^^^^^^^^^

Use of `~.Axes.plot_date` has been discouraged since Matplotlib 3.5 and the function is
now formally deprecated.

- ``datetime``-like data should directly be plotted using `~.Axes.plot`.
- If you need to plot plain numeric data as :ref:`date-format` or need to set a
  timezone, call ``ax.xaxis.axis_date`` / ``ax.yaxis.axis_date`` before `~.Axes.plot`.
  See `.Axis.axis_date`.

Legend labels for ``plot``
^^^^^^^^^^^^^^^^^^^^^^^^^^

Previously if a sequence was passed to the *label* parameter of `~.Axes.plot` when
plotting a single dataset, the sequence was automatically cast to string for the legend
label. This behavior is now deprecated and in future will error if the sequence length
is not one (consistent with multi-dataset behavior, where the number of elements must
match the number of datasets). To keep the old behavior, cast the sequence to string
before passing.

``boxplot`` tick labels
^^^^^^^^^^^^^^^^^^^^^^^

The parameter *labels* has been renamed to *tick_labels* for clarity and consistency
with `~.Axes.bar`.

Mixing positional and keyword arguments for ``legend`` handles and labels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This previously only raised a warning, but is now formally deprecated. If passing
*handles* and *labels*, they must be passed either both positionally or both as keyword.

Applying theta transforms in ``PolarTransform``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Applying theta transforms in `~matplotlib.projections.polar.PolarTransform` and
`~matplotlib.projections.polar.InvertedPolarTransform` is deprecated, and will be
removed in a future version of Matplotlib. This is currently the default behaviour when
these transforms are used externally, but only takes affect when:

- An axis is associated with the transform.
- The axis has a non-zero theta offset or has theta values increasing in a clockwise
  direction.

To silence this warning and adopt future behaviour, set
``apply_theta_transforms=False``. If you need to retain the behaviour where theta values
are transformed, chain the ``PolarTransform`` with a `~matplotlib.transforms.Affine2D`
transform that performs the theta shift and/or sign shift.

*interval* parameter of ``TimerBase.start``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Setting the timer *interval* while starting it is deprecated. The interval can be
specified instead in the timer constructor, or by setting the ``timer.interval``
attribute.

*nth_coord* parameter to axisartist helpers for fixed axis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Helper APIs in `.axisartist` for generating a "fixed" axis on rectilinear axes
(`.FixedAxisArtistHelperRectilinear`) no longer take a *nth_coord* parameter, as that
parameter is entirely inferred from the (required) *loc* parameter and having
inconsistent *nth_coord* and *loc* is an error.

For curvilinear axes, the *nth_coord* parameter remains supported (it affects the
*ticks*, not the axis position itself), but that parameter will become keyword-only, for
consistency with the rectilinear case.

``rcsetup.interactive_bk``, ``rcsetup.non_interactive_bk`` and ``rcsetup.all_backends``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

... are deprecated and replaced by ``matplotlib.backends.backend_registry.list_builtin``
with the following arguments

- ``matplotlib.backends.BackendFilter.INTERACTIVE``
- ``matplotlib.backends.BackendFilter.NON_INTERACTIVE``
- ``None``

respectively.

Miscellaneous deprecations
^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``backend_ps.get_bbox_header`` is considered an internal helper
- ``BboxTransformToMaxOnly``; if you rely on this, please make a copy of the code
- ``ContourLabeler.add_label_clabeltext``
- ``TransformNode.is_bbox``; instead check the object using ``isinstance(...,
  BboxBase)``
- ``GridHelperCurveLinear.get_tick_iterator``
