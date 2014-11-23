Removed `lod` from Artist
`````````````````````````

Removed the method *set_lod* and all references to
the attribute *_lod* as the are not used anywhere else in the
code base.  It appears to be a feature stub that was never built
out.
