`color_cycle` deprecated
````````````````````````

In light of the new property cycling feature,
the Axes method *set_color_cycle* is now deprecated.
Calling this method will replace the current property cycle with
one that cycles just the given colors.

Similarly, the rc parameter *axes.color_cycle* is also deprecated in
lieu of the new *axes.prop_cycle* parameter. Having both parameters in
the same rc file is not recommended as the result cannot be
predicted. For compatibility, setting *axes.color_cycle* will
replace the cycler in *axes.prop_cycle* with a color cycle.
Accessing *axes.color_cycle* will return just the color portion
of the property cycle, if it exists.

Timeline for removal has not been set.
