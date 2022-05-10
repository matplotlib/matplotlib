stairs(..., fill=True) to hide patch edge by setting lw=0
---------------------------------------------------------

``stairs(..., fill=True)`` would previously hide Patch
edge by setting edgecolor="none". Calling ``set_color()``
on the Patch after would make the Patch appear larger.
Updated treatment prevents this. Likewise calling
``stairs(..., fill=True, lw=3)`` will behave more
transparently.
