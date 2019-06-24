Improvements in Logit scale ticker and formatter
------------------------------------------------

Introduced in version 1.5, the logit scale didn't have appropriate ticker and
formatter. Previously, location of ticks was not zoom dependent, too many label
was displayed implying overlapping which break readability, and label formatting
was not precision adaptive.

Starting from this version, the locator have near the same behavior as the
locator for the log scale or the same behavior as the locator for the linear
scale, depending on used zoom. The number of ticks is controlled. Some minor
labels are displayed adaptively as sublabels in log scale. Formatting is adapted
for probabilities and the precision is adaptive depending on the scale.
