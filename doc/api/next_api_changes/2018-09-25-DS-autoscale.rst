Autoscaling when no data present
--------------------------------

`.Axes.autoscale_view` now does not attempt to autoscale an axis if there is no
data with finite coordinates present to use to determine the autoscaling.
