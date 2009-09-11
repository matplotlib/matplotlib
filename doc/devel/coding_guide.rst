.. _coding-guide:

************
Coding guide
************

.. _version-control:

Version control
===============

.. _using-svn:

svn checkouts
-------------

Checking out everything in the trunk (matplotlib and toolkits)::

   svn co https://matplotlib.svn.sourceforge.net/svnroot/matplotlib/trunk \
   matplotlib --username=youruser --password=yourpass

Checking out the main source::

   svn co https://matplotlib.svn.sourceforge.net/svnroot/matplotlib/trunk/\
   matplotlib mpl --username=youruser --password=yourpass

Branch checkouts, eg the release branch::

   svn co https://matplotlib.svn.sf.net/svnroot/matplotlib/branches/v0_99_maint mpl99


Committing changes
------------------

When committing changes to matplotlib, there are a few things to bear
in mind.

* if your changes are non-trivial, please make an entry in the
  :file:`CHANGELOG`

* if you change the API, please document it in :file:`doc/api/api_changes.rst`,
  and consider posting to `matplotlib-devel
  <http://lists.sourceforge.net/mailman/listinfo/matplotlib-devel>`_

* Are your changes python2.4 compatible?  We still support 2.4, so
  avoid features new to 2.5

* Can you pass :file:`examples/tests/backend_driver.py`?  This is our
  poor man's unit test.

* Can you add a test to :file:`unit/nose_tests.py` to test your changes?

* If you have altered extension code, do you pass
  :file:`unit/memleak_hawaii.py`?

* if you have added new files or directories, or reorganized existing
  ones, are the new files included in the match patterns in
  :file:`MANIFEST.in`.  This file determines what goes into the source
  distribution of the mpl build.

* Keep the release branch (eg 0.90 and trunk in sync where it makes
  sense.  If there is a bug on both that needs fixing, use
  `svnmerge.py <http://www.orcaware.com/svn/wiki/Svnmerge.py>`_ to
  keep them in sync.  See :ref:`svn-merge` below.

.. _svn-merge:

Using svnmerge
--------------

svnmerge is useful for making bugfixes to a maintenance branch, and
then bringing those changes into the trunk.

The basic procedure is:

* install ``svnmerge.py`` in your PATH::

    > wget http://svn.collab.net/repos/svn/trunk/contrib/client-side/\
      svnmerge/svnmerge.py

* get a svn checkout of the branch you'll be making bugfixes to and
  the trunk (see above)

* Create and commit the bugfix on the branch.

* Then make sure you svn upped on the trunk and have no local
  modifications, and then from your checkout of the svn trunk do::

       svnmerge.py merge -S BRANCHNAME

  Where BRANCHNAME is the name of the branch to merge *from*,
  e.g. v0_99_maint.

  If you wish to merge only specific revisions (in an unusual
  situation), do::

      > svnmerge.py merge -rNNN1-NNN2

  where the ``NNN`` are the revision numbers.  Ranges are also
  acceptable.

  The merge may have found some conflicts (code that must be manually
  resolved).  Correct those conflicts, build matplotlib and test your
  choices.  If you have resolved any conflicts, you can let svn clean
  up the conflict files for you::

      > svn -R resolved .

  ``svnmerge.py`` automatically creates a file containing the commit
  messages, so you are ready to make the commit::

     > svn commit -F svnmerge-commit-message.txt


.. _setting-up-svnmerge:

Setting up svnmerge
~~~~~~~~~~~~~~~~~~~

.. note::
   The following applies only to release managers when there is
   a new release.  Most developers will not have to concern themselves
   with this.

* Creating a new branch from the trunk (if the release version is
  0.98.5 at revision 6573)::

      > svn copy \
      https://matplotlib.svn.sf.net/svnroot/matplotlib/trunk/matplotlib@6573 \
      https://matplotlib.svn.sf.net/svnroot/matplotlib/branches/v0_98_5_maint \
      -m "Creating maintenance branch for 0.98.5"

* You can add a new branch for the trunk to "track" using
  "svnmerge.py init", e.g., from a working copy of the trunk::

      > svnmerge.py init https://matplotlib.svn.sourceforge.net/svnroot/matplotlib/branches/v0_98_5_maint
      property 'svnmerge-integrated' set on '.'

  After doing a "svn commit" on this, this merge tracking is available
  to everyone, so there's no need for anyone else to do the "svnmerge
  init".

* Tracking can later be removed with the "svnmerge.py uninit" command,
  e.g.::

      > svnmerge.py -S v0_9_5_maint uninit

.. _using-git:

Using git
---------

Some matplotlib developers are experimenting with using git on top of
the subversion repository.  Developers are not required to use git, as
subversion will remain the canonical central repository for the
foreseeable future.

Cloning the git mirror
~~~~~~~~~~~~~~~~~~~~~~

There is an experimental `matplotlib github mirror`_ of the subversion
repository. To make a local clone of it in the directory ``mpl.git``,
enter the following commands::

  # This will create your copy in the mpl.git directory
  git clone git://github.com/astraw/matplotlib.git mpl.git
  cd mpl.git
  git config --add remote.origin.fetch +refs/remotes/*:refs/remotes/*
  git fetch
  git svn init --branches=branches --trunk=trunk/matplotlib --tags=tags https://matplotlib.svn.sourceforge.net/svnroot/matplotlib

  # Now just get the latest svn revisions from the SourceForge SVN repository
  git svn fetch -r 6800:HEAD

.. _matplotlib github mirror: http://github.com/astraw/matplotlib

To install from this cloned repository, use the commands in the
:ref:`svn installation <install-svn>` section::

  > cd mpl.git
  > python setup.py install

Using git
~~~~~~~~~

The following is a suggested workflow for git/git-svn.

Start with a virgin tree in sync with the svn trunk on the git branch
"master"::

  git checkout master
  git svn rebase

To create a new, local branch called "whizbang-branch"::

  git checkout -b whizbang-branch

Do make commits to the local branch::

  # hack on a bunch of files
  git add bunch of files
  git commit -m "modified a bunch of files"
  # repeat this as necessary

Now, go back to the master branch and append the history of your branch
to the master branch, which will end up as the svn trunk::

  git checkout master
  git svn rebase # Ensure we have most recent svn
  git rebase whizbang-branch # Append whizbang changes to master branch
  git svn dcommit -n # Check that this will apply to svn
  git svn dcommit # Actually apply to svn

Finally, you may want to continue working on your whizbang-branch, so
rebase it to the new master::

  git checkout whizbang-branch
  git rebase master

If you get the dreaded "Unable to determine upstream SVN information
from working tree history" error when running "git svn rebase", try
creating a new git branch based on subversion trunk and cherry pick
your patches onto that::

  git checkout -b work remotes/trunk # create a new "work" branch
  git cherry-pick <commit> # where <commit> will get applied to new branch

Working on a maintenance branch from git
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The matplotlib maintenance branches are also available through git.
(Note that the ``git svn init`` line in the instructions above was
updated to make this possible.  If you created your git mirror without
a ``--branches`` option, you will need to perform all of the steps
again in a new directory).

You can see which branches are available with::

  git branch -a

To switch your working copy to the 0.98.5 maintenance branch::

  git checkout v0_98_5_maint

Then you probably want to (as above) create a new local branch based
on that branch::

  git checkout -b whizbang-branch

When you ``git svn dcommit`` from a maintenance branch, it will commit
to that branch, not to the trunk.

While it should theoretically be possible to perform merges from a git
maintenance branch to a git trunk and then commit those changes back
to the SVN trunk, I have yet to find the magic incantation to make
that work.  However, svnmerge as described `above <svn-merge>`_ can be
used and in fact works quite well.

A note about git write access
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The matplotlib developers need to figure out if there should be write
access to the git repository. This implies using the personal URL
(``git@github.com:astraw/matplotlib.git``) rather than the public URL
(``git://github.com/astraw/matplotlib.git``) for the
repository. However, doing so may make life complicated in the sense
that then there are two writeable matplotlib repositories, which must
be synced to prevent divergence. This is probably not an
insurmountable problem, but it is a problem that the developers should
reach a consensus about. Watch this space...

.. _style-guide:

Style guide
===========

Importing and name spaces
-------------------------

For `numpy <http://www.numpy.org>`_, use::

  import numpy as np
  a = np.array([1,2,3])

For masked arrays, use::

  import numpy.ma as ma

For matplotlib main module, use::

  import matplotlib as mpl
  mpl.rcParams['xtick.major.pad'] = 6

For matplotlib modules (or any other modules), use::

  import matplotlib.cbook as cbook

  if cbook.iterable(z):
      pass

We prefer this over the equivalent ``from matplotlib import cbook``
because the latter is ambiguous as to whether ``cbook`` is a module or a
function.  The former makes it explicit that you
are importing a module or package.  There are some modules with names
that match commonly used local variable names, eg
:mod:`matplotlib.lines` or :mod:`matplotlib.colors`. To avoid the clash,
use the prefix 'm' with the ``import some.thing as
mthing`` syntax, eg::

    import matplotlib.lines as mlines
    import matplotlib.transforms as transforms   # OK
    import matplotlib.transforms as mtransforms  # OK, if you want to disambiguate
    import matplotlib.transforms as mtrans       # OK, if you want to abbreviate

Naming, spacing, and formatting conventions
-------------------------------------------

In general, we want to hew as closely as possible to the standard
coding guidelines for python written by Guido in `PEP 0008
<http://www.python.org/dev/peps/pep-0008>`_, though we do not do this
throughout.

* functions and class methods: ``lower`` or
  ``lower_underscore_separated``

* attributes and variables: ``lower`` or ``lowerUpper``

* classes: ``Upper`` or ``MixedCase``

Prefer the shortest names that are still readable.

Configure your editor to use spaces, not hard tabs. The standard
indentation unit is always four spaces;
if there is a file with
tabs or a different number of spaces it is a bug -- please fix it.
To detect and fix these and other whitespace errors (see below),
use `reindent.py
<http://svn.python.org/projects/doctools/trunk/utils/reindent.py>`_ as
a command-line script.  Unless you are sure your editor always
does the right thing, please use reindent.py before checking changes into
svn.

Keep docstrings_ uniformly indented as in the example below, with
nothing to the left of the triple quotes.  The
:func:`matplotlib.cbook.dedent` function is needed to remove excess
indentation only if something will be interpolated into the docstring,
again as in the example below.

Limit line length to 80 characters.  If a logical line needs to be
longer, use parentheses to break it; do not use an escaped newline.
It may be preferable to use a temporary variable to replace a single
long line with two shorter and more readable lines.

Please do not commit lines with trailing white space, as it causes
noise in svn diffs.  Tell your editor to strip whitespace from line
ends when saving a file.  If you are an emacs user, the following in
your ``.emacs`` will cause emacs to strip trailing white space upon
saving for python, C and C++:

.. code-block:: cl

  ; and similarly for c++-mode-hook and c-mode-hook
  (add-hook 'python-mode-hook
            (lambda ()
            (add-hook 'write-file-functions 'delete-trailing-whitespace)))

for older versions of emacs (emacs<22) you need to do:

.. code-block:: cl

  (add-hook 'python-mode-hook
            (lambda ()
            (add-hook 'local-write-file-hooks 'delete-trailing-whitespace)))

Keyword argument processing
---------------------------

Matplotlib makes extensive use of ``**kwargs`` for pass-through
customizations from one function to another.  A typical example is in
:func:`matplotlib.pylab.text`.  The definition of the pylab text
function is a simple pass-through to
:meth:`matplotlib.axes.Axes.text`::

  # in pylab.py
  def text(*args, **kwargs):
      ret =  gca().text(*args, **kwargs)
      draw_if_interactive()
      return ret

:meth:`~matplotlib.axes.Axes.text` in simplified form looks like this,
i.e., it just passes all ``args`` and ``kwargs`` on to
:meth:`matplotlib.text.Text.__init__`::

  # in axes.py
  def text(self, x, y, s, fontdict=None, withdash=False, **kwargs):
      t = Text(x=x, y=y, text=s, **kwargs)

and :meth:`~matplotlib.text.Text.__init__` (again with liberties for
illustration) just passes them on to the
:meth:`matplotlib.artist.Artist.update` method::

  # in text.py
  def __init__(self, x=0, y=0, text='', **kwargs):
      Artist.__init__(self)
      self.update(kwargs)

``update`` does the work looking for methods named like
``set_property`` if ``property`` is a keyword argument.  I.e., no one
looks at the keywords, they just get passed through the API to the
artist constructor which looks for suitably named methods and calls
them with the value.

As a general rule, the use of ``**kwargs`` should be reserved for
pass-through keyword arguments, as in the example above.  If all the
keyword args are to be used in the function, and not passed
on, use the key/value keyword args in the function definition rather
than the ``**kwargs`` idiom.

In some cases, you may want to consume some keys in the local
function, and let others pass through.  You can ``pop`` the ones to be
used locally and pass on the rest.  For example, in
:meth:`~matplotlib.axes.Axes.plot`, ``scalex`` and ``scaley`` are
local arguments and the rest are passed on as
:meth:`~matplotlib.lines.Line2D` keyword arguments::

  # in axes.py
  def plot(self, *args, **kwargs):
      scalex = kwargs.pop('scalex', True)
      scaley = kwargs.pop('scaley', True)
      if not self._hold: self.cla()
      lines = []
      for line in self._get_lines(*args, **kwargs):
          self.add_line(line)
          lines.append(line)

Note: there is a use case when ``kwargs`` are meant to be used locally
in the function (not passed on), but you still need the ``**kwargs``
idiom.  That is when you want to use ``*args`` to allow variable
numbers of non-keyword args.  In this case, python will not allow you
to use named keyword args after the ``*args`` usage, so you will be
forced to use ``**kwargs``.  An example is
:meth:`matplotlib.contour.ContourLabeler.clabel`::

  # in contour.py
  def clabel(self, *args, **kwargs):
      fontsize = kwargs.get('fontsize', None)
      inline = kwargs.get('inline', 1)
      self.fmt = kwargs.get('fmt', '%1.3f')
      colors = kwargs.get('colors', None)
      if len(args) == 0:
          levels = self.levels
          indices = range(len(self.levels))
      elif len(args) == 1:
         ...etc...

.. _docstrings:

Documentation and docstrings
============================

Matplotlib uses artist introspection of docstrings to support
properties.  All properties that you want to support through ``setp``
and ``getp`` should have a ``set_property`` and ``get_property``
method in the :class:`~matplotlib.artist.Artist` class.  Yes, this is
not ideal given python properties or enthought traits, but it is a
historical legacy for now.  The setter methods use the docstring with
the ACCEPTS token to indicate the type of argument the method accepts.
Eg. in :class:`matplotlib.lines.Line2D`::

  # in lines.py
  def set_linestyle(self, linestyle):
      """
      Set the linestyle of the line

      ACCEPTS: [ '-' | '--' | '-.' | ':' | 'steps' | 'None' | ' ' | '' ]
      """

Since matplotlib uses a lot of pass-through ``kwargs``, eg. in every
function that creates a line (:func:`~matplotlib.pyplot.plot`,
:func:`~matplotlib.pyplot.semilogx`,
:func:`~matplotlib.pyplot.semilogy`, etc...), it can be difficult for
the new user to know which ``kwargs`` are supported.  Matplotlib uses
a docstring interpolation scheme to support documentation of every
function that takes a ``**kwargs``.  The requirements are:

1. single point of configuration so changes to the properties don't
   require multiple docstring edits.

2. as automated as possible so that as properties change, the docs
   are updated automagically.

The functions :attr:`matplotlib.artist.kwdocd` and
:func:`matplotlib.artist.kwdoc` to facilitate this.  They combine
python string interpolation in the docstring with the matplotlib
artist introspection facility that underlies ``setp`` and ``getp``.
The ``kwdocd`` is a single dictionary that maps class name to a
docstring of ``kwargs``.  Here is an example from
:mod:`matplotlib.lines`::

  # in lines.py
  artist.kwdocd['Line2D'] = artist.kwdoc(Line2D)

Then in any function accepting :class:`~matplotlib.lines.Line2D`
pass-through ``kwargs``, eg. :meth:`matplotlib.axes.Axes.plot`::

  # in axes.py
  def plot(self, *args, **kwargs):
      """
      Some stuff omitted

      The kwargs are Line2D properties:
      %(Line2D)s

      kwargs scalex and scaley, if defined, are passed on
      to autoscale_view to determine whether the x and y axes are
      autoscaled; default True.  See Axes.autoscale_view for more
      information
      """
      pass
  plot.__doc__ = cbook.dedent(plot.__doc__) % artist.kwdocd

Note there is a problem for :class:`~matplotlib.artist.Artist`
``__init__`` methods, eg. :meth:`matplotlib.patches.Patch.__init__`,
which supports ``Patch`` ``kwargs``, since the artist inspector cannot
work until the class is fully defined and we can't modify the
``Patch.__init__.__doc__`` docstring outside the class definition.
There are some some manual hacks in this case, violating the
"single entry point" requirement above -- see the
``artist.kwdocd['Patch']`` setting in :mod:`matplotlib.patches`.

.. _custom_backend:

Developing a new backend
========================

If you are working on a custom backend, the *backend* setting in
:file:`matplotlibrc` (:ref:`customizing-matplotlib`) supports an
external backend via the ``module`` directive.  if
:file:`my_backend.py` is a matplotlib backend in your
:envvar:`PYTHONPATH`, you can set use it on one of several ways

* in matplotlibrc::

    backend : module://my_backend

* with the use directive is your script::

    import matplotlib
    matplotlib.use('module://my_backend')

* from the command shell with the -d flag::

    > python simple_plot.py -d module://my_backend



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
are acceptable in matplotlib toolkits (eg basemap), but make sure you
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
license. I.e., you are required to give the source code to other
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
