Slider UI widget can snap to discrete values
--------------------------

The slider UI widget can take the optional argument `valstep`.  Doing so
forces the slider to take on only discrete values, starting from `valmin` and
counting up to `valmax` with steps of size `valstep`.

If `closedmax==True`, then the slider will snap to `valmax` as well.  
