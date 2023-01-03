The object returned by ``pcolor()`` has changed to a ``PolyQuadMesh`` class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The old object was a `.PolyCollection` with flattened vertices and array data.
The new `.PolyQuadMesh` class subclasses `.PolyCollection`, but adds in better
2D coordinate and array handling in alignment with `.QuadMesh`. Previously, if
a masked array was input, the list of polygons within the collection would shrink
to the size of valid polygons and users were required to keep track of which
polygons were drawn and call ``set_array()`` with the smaller "compressed" array size.
Passing the "compressed" and flattened array values is now deprecated and the
full 2D array of values (including the mask) should be passed
to `.PolyQuadMesh.set_array`.
