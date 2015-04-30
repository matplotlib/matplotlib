Add salt to cilpPath id
```````````````````````

Add salt to the hash used to determine the id of the ``clipPath``
nodes.  This is to avoid conflicts in two svg documents with the same
clip path are included in the same document (see
https://github.com/ipython/ipython/issues/8133 and
https://github.com/matplotlib/matplotlib/issues/4349 ), however this
means that the svg output is no longer deterministic if the same
figure is saved twice.  It is not expected that this will affect any
users as the current ids are generated from an md5 hash of properties
of the clip path and any user would have a very difficult time
anticipating the value of the id.
