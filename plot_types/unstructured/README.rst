.. _unstructured_plots:

Unstructured coordinates
-------------------------

Sometimes we collect data ``z`` at coordinates ``(x,y)`` and want to visualize
as a contour.  Instead of gridding the data and then using 
`~.axes.Axes.contour`, we can use a triangulation algorithm and fill the 
triangles.