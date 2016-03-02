.. _license-discussion:

Licenses
========

Matplotlib only uses BSD compatible code.  If you bring in code from
another project make sure it has a PSF, BSD, MIT or compatible license
(see the Open Source Initiative `licenses page
<http://www.opensource.org/licenses>`_ for details on individual
licenses).  If it doesn't, you may consider contacting the author and
asking them to relicense it.  GPL and LGPL code are not acceptable in
the main code base, though we are considering an alternative way of
distributing L/GPL code through an separate channel, possibly a
toolkit.  If you include code, make sure you include a copy of that
code's license in the license directory if the code's license requires
you to distribute the license with it.  Non-BSD compatible licenses
are acceptable in matplotlib toolkits (e.g., basemap), but make sure you
clearly state the licenses you are using.

Why BSD compatible?
-------------------

The two dominant license variants in the wild are GPL-style and
BSD-style. There are countless other licenses that place specific
restrictions on code reuse, but there is an important difference to be
considered in the GPL and BSD variants.  The best known and perhaps
most widely used license is the GPL, which in addition to granting you
full rights to the source code including redistribution, carries with
it an extra obligation. If you use GPL code in your own code, or link
with it, your product must be released under a GPL compatible
license. i.e., you are required to give the source code to other
people and give them the right to redistribute it as well. Many of the
most famous and widely used open source projects are released under
the GPL, including linux, gcc, emacs and sage.

The second major class are the BSD-style licenses (which includes MIT
and the python PSF license). These basically allow you to do whatever
you want with the code: ignore it, include it in your own open source
project, include it in your proprietary product, sell it,
whatever. python itself is released under a BSD compatible license, in
the sense that, quoting from the PSF license page::

    There is no GPL-like "copyleft" restriction. Distributing
    binary-only versions of Python, modified or not, is allowed. There
    is no requirement to release any of your source code. You can also
    write extension modules for Python and provide them only in binary
    form.

Famous projects released under a BSD-style license in the permissive
sense of the last paragraph are the BSD operating system, python and
TeX.

There are several reasons why early matplotlib developers selected a
BSD compatible license. matplotlib is a python extension, and we
choose a license that was based on the python license (BSD
compatible).  Also, we wanted to attract as many users and developers
as possible, and many software companies will not use GPL code in
software they plan to distribute, even those that are highly committed
to open source development, such as `enthought
<http://enthought.com>`_, out of legitimate concern that use of the
GPL will "infect" their code base by its viral nature. In effect, they
want to retain the right to release some proprietary code. Companies
and institutions who use matplotlib often make significant
contributions, because they have the resources to get a job done, even
a boring one. Two of the matplotlib backends (FLTK and WX) were
contributed by private companies.  The final reason behind the
licensing choice is compatibility with the other python extensions for
scientific computing: ipython, numpy, scipy, the enthought tool suite
and python itself are all distributed under BSD compatible licenses.
