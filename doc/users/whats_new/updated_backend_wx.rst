wx backend has been updated
---------------------------
The wx backend can now be used with both wxPython classic and
`Phoenix <http://wxpython.org/Phoenix/docs/html/main.html>`__.

wxPython classic has to be at least version 2.8.12 and works on Python 2.x,
wxPython Phoenix needs a current snapshot and works on Python 2.7 and 3.4+.

If you have multiple versions of wxPython installed, then the user code is
responsible to set the wxPython version you want to use.  How to do this is
explained in the comment at the beginning of example the 
`examples\user_interfaces\embedding_in_wx2.py`.
