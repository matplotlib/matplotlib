``Artist`` gained setter and getter for new ``_in_autoscale`` flag
-------------------------------------------------------------------

The ``_in_autoscale`` flag determines whether the instance is used
in the autoscale calculation. The flag can be a bool, or tuple[bool] for 2D/3D.
Expansion to a tuple is done in the setter.

The purpose is to put auto-limit logic inside respective Artists.
This allows ``Collection`` objects to be used in ``relim`` axis calculations.
