Pausing and Resuming Animations
-------------------------------
We have added public `pause()` and `resume()` methods to the base
matplotlib.animation.Animation class that allows you to, well, pause
and resume animations. These methods can be used as callbacks for event
listeners on UI elements so that your plots can have some playback
control UI.
