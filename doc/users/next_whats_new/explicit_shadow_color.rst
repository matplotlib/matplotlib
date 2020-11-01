Legend shadow colors can be explicitly defined
----------------------------------------------

The shadow parameter for the Legend constructor now
accepts both color and bool. If it is a bool, the
behavior is exactly the same as before.
Any colorlike value sets the shadow to that color,
subject to the same transparency effect that the default
facecolor undergoes.

An accessory validation function `~.rcsetup.validate_color_or_bool`
has also been added, which preferentially returns a color.