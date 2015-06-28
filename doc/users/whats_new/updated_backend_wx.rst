wx backend has been updated
---------------------------

The wx backend can now be used with both wxPython classic and
`Phoenix <http://wxpython.org/Phoenix/docs/html/main.html>`__.

wxPython classic has to be at least version 2.8.12 and works on Python 2.x. As
of May 2015 no official release of wxPython Phoenix is available but a
current snapshot will work on Python 2.7+ and 3.4+.

If you have multiple versions of wxPython installed, then the user code is
responsible setting the wxPython version.  How to do this is
explained in the comment at the beginning of the example
`examples\user_interfaces\embedding_in_wx2.py`.
