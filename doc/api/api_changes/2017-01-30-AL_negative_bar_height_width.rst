`bar` now returns rectangles of negative height or width if the corresponding input is negative
```````````````````````````````````````````````````````````````````````````````````````````````

`plt.bar` used to normalize the coordinates of the rectangles that it created,
to keep their height and width positives, even if the corresponding input was
negative.  This normalization has been removed to permit a simpler computation
of the correct `sticky_edges` to use.
