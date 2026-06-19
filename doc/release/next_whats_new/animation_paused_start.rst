Animations can now be created in a paused state
--------------------------------------------------------------------------------
'.Animation' now accepts a 'paused' keyword argument. When set to 'True', the animation will not automatically start when the figure is first drawn.
This allows starting an animation in a paused state without needing to access private API. Additionally, '.TimerBase.is_running' was added to query whether the underlying event source is currently running.
::
    ani = FuncAnimation(fig, update, frames=10, paused=True)
    ani.event_source.is_running()  # False
    ani.resume()
    ani.event_source.is_running()  # True