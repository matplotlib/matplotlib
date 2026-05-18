``Path.arc`` can opt out of angular unwrapping
----------------------------------------------
`~matplotlib.path.Path.arc` now accepts an *unwrap_angles* keyword-only
parameter. The default value ``True`` preserves the previous behaviour
of collapsing requests for arcs spanning more than 360 degrees to the
shortest equivalent arc. Passing ``unwrap_angles=False`` honours the
caller's exact angular span.

The primary motivation is the floating-point edge case where a delta
of nearly-but-not-exactly 360 degrees was unwrapped to a near-empty
arc, which is the root cause of the polar gridline and outer-spine
collapse fixed in this release. The parameter also lets callers
generate multi-turn arcs (spans greater than 360 degrees) directly.
