Improved Delaunay triangulations with large offsets
```````````````````````````````````````````````````

Delaunay triangulations now deal with large x,y offsets in a better
way. This can cause minor changes to any triangulations calculated
using Matplotlib, i.e. any use of `matplotlib.tri.Triangulation` that
requests that a Delaunay triangulation is calculated, which includes
`matplotlib.pyplot.tricontour`, `matplotlib.pyplot.tricontourf`,
`matplotlib.pyplot.tripcolor`, `matplotlib.pyplot.triplot`,
`mlab.griddata` and `mpl_toolkits.mplot3d.plot_trisurf`.
