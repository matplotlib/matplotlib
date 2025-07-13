Streamplot integration control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two new options have been added to the `~.axes.Axes.streamplot` function that
give the user better control of the streamline integration. The first is called
``integration_max_step_scale`` and multiplies the default max step computed by the
integrator. The second is called ``integration_max_error_scale`` and multiplies the
default max error set by the integrator. Values for these parameters between
zero and one reduce (tighten) the max step or error to improve streamline
accuracy by performing more computation. Values greater than one increase
(loosen) the max step or error to reduce computation time at the cost of lower
streamline accuracy.

The integrator defaults are both hand-tuned values and may not be applicable to
all cases, so this allows customizing the behavior to specific use cases.
Modifying only ``integration_max_step_scale`` has proved effective, but it may be useful
to control the error as well.
