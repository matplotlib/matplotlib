Elliptical arcs now drawn between correct angles
````````````````````````````````````````````````

The `matplotlib.patches.Arc` patch is now correctly drawn between the given
angles.

Previously a circular arc was drawn and then stretched into an ellipse,
so the resulting arc did not lie between *theta1* and *theta2*.
