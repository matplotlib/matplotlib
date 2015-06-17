deprecated idle_event
`````````````````````

The `idle_event` was broken or missing in most backends and causes spurious
warnings in some cases, and its use in creating animations is now obsolete due
to the animations module. Therefore code involving it has been removed from all
but the wx backend (where it partially works), and its use is deprecated.  The
animations module may be used instead to create animations.

