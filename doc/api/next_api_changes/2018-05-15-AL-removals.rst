Changes regarding the text.latex.unicode rcParam
````````````````````````````````````````````````

The rcParam now defaults to True and is deprecated (i.e., in future versions
of Maplotlib, unicode input will always be supported).

Moreover, the underlying implementation now uses ``\usepackage[utf8]{inputenc}``
instead of ``\usepackage{ucs}\usepackage[utf8x]{inputenc}``.
