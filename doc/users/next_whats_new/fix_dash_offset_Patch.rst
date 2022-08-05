Fix the dash offset of the Patch class
--------------------------------------
Traditionally, when setting the linestyle on a `.Patch` object using a dash tuple the
offset was ignored. Now the offset is passed to the draw method of Patch as expected
and it can be used as it is used with Line2D objects.
