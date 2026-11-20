DVI Parsing enhancements
------------------------

Matplotlib is capable of reading ``.dvi`` files with `.dviread.Dvi`, which has historically worked well for its existing use cases, but did not provide the granularity to inspect the raw DVI operations in a file, and didn't have a way to report color information upwards to the various backends that might care about color directives.

The new `.dviread.Ops` namespace provides the ability to inspect a DVI file one op at a time, `.dviread.VM` handles state tracking (and can be driven manually with its ``.op_foo(code, **args)`` methods, and the ``.dviread.Text`` and ``.dviread.Box`` classes have been modified to store color information in a backwards-compatible way.

While backends don't render color directives yet, this important groundwork lets them *see* color directives, so that they can be acted on in the future.

>>> import matplotlib.dviread as dr
>>> for op in dr.Ops.read_file("./some/document.dvi"):
...     print(op)
...
>>> for page in dr.Dvi("./some/document.dvi", 72):
...     for t in page.text:
...         print(t.glyph, t.color)
