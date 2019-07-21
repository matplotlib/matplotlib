Improvements in Logit scale ticker and formatter
------------------------------------------------

Introduced in version 1.5, the logit scale didn't have an appropriate ticker and
formatter. Previously, the location of ticks was not zoom dependent, too many labels
were displayed causing overlapping which broke readability, and label formatting
did not adapt to precision.

Starting from this version, the logit locator has nearly the same behavior as the
locator for the log scale or the linear
scale, depending on used zoom. The number of ticks is controlled. Some minor
labels are displayed adaptively as sublabels in log scale. Formatting is adapted
for probabilities and the precision is adapts to the scale.
