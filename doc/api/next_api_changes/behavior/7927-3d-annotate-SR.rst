Axes3D annotations return Annotation3D
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`~mpl_toolkits.mplot3d.axes3d.Axes3D.annotate` now constructs and returns an
`~mpl_toolkits.mplot3d.art3d.Annotation3D` instance. Previously it returned a
2D `.Annotation` that was then mutated in-place to add 3D behavior. The new
behavior keeps all 3D validation and projection logic inside `Annotation3D`
while preserving the public API and semantics of `.Axes.annotate`.
