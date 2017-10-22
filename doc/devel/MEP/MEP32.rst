MEP32: ``matplotlibrc``/``mplstyle`` with Python syntax
=======================================================

.. contents:: :local:

Status
------

Discussion

Branches and Pull Requests
--------------------------

None

Abstract
--------

I propose to replace ``matplotlibrc`` and ``mplstyle`` files (henceforth
"old-style configs") by files using regular Python syntax (henceforth
"new-style configs").  This will fix a number of issues that currently exist
with parsing old-style configs.

Detailed description
--------------------

The problem
~~~~~~~~~~~

Current ("old-style") configuration files use a custom syntax, of the form

.. code:: conf

   key: value  # possible comment
   key: value

For each allowable key, a specific parser is defined in ``mpl.rcsetup``,
possibly via some helper functions.  Typical parsers parse booleans, numbers,
strings (always unquoted), enumerated strings, comma-separated lists of the
above-mentioned types, etc.

However, some desirable inputs are difficult to parse, or currently only
partially parsed:

- Property cycles (e.g., ``axes.prop_cycle``), of the form ::

     cycler("key", [value1, value2, ...])

  are currently simply parsed via ``eval`` in a restricted environment
  (`#6274`_), which may (or may not) be a security hole (especially combined
  with the ability to load style files from an URL).

- Path effects (e.g., for implementing an XKCD style sheet, `#6157`_).  They
  would have a form similar to ::

      patheffects.withStroke(linewidth=4, foreground="w")

  and would, like ``cycler``\s, require either a custom parser or being
  ``eval``'d.

- Long inputs (e.g. property cycles) cannot be split over multiple lines, as
  the parser has no support for line continuations (`#9184`_).

- LaTeX and PGF preambles (multiple, comma separated strings) cannot contain
  commas, because commas are replaced by newlines by the parser (`#4371`_).
  Note that commas are in fact fairly common in "normal" LaTeX preambles (e.g.,
  ``\usepackage[option1, option2]{package}``.  Actually inputting the preamble
  over multiple lines is not possible due to the lack of multiline support (see
  above).

- Strings cannot contain the hash (``#``) symbol, as strings are unquoted and
  the hash is unconditionally interpreted as the start of a comment (`#7089`_).
  The hash symbol had been proposed to indicate a "plus" marker.

- Dash specs of the form ``(offset, (ink-on, ink-off, in-on, ...))`` are
  misparsed: while ::

     plt.plot([1, 2], ls=(0, (5, 5)))

  works just fine, ::

     plt.rcParams["lines.linestyle"] = (0, (5, 5))

  and ::

     plt.rcParams["lines.linestyle"] = "(0, (5, 5))"

  as well as setting this value in ``matplotlibrc`` all raise an exception
  (indirectly a cause of `#7219`_).

- Custom color palettes (redefining the meaning of ``"r"``, ``"g"``, ``"b"``,
  etc. as seaborn used to do) has been proposed (`#8430`_), but the
  hypothetical rcParam value would have a type of ``Dict[str, Dict[str, str]]``
  (mapping palettes to mappings of color names to color values), which was
  described by @tacaswell as "not too hard to parse" but "would further stress
  our current configuration system".

Overall, the syntax of the config file is defined as "whatever the parser
accepts" (`#3670`_).

An additional feature that has been requested (but shoudl not be particularly
difficult to implement using the current machinery) is "cascading" style
sheets, either by adding a ``style`` key to style files (`#4240`_) or by
loading all ``matplotlibrc``\s in order (`#6320`_).

Proposed solution
~~~~~~~~~~~~~~~~~

Instead of playing whack-a-mole with parser bugs, I propose to replace the
syntax of config files to simply use Python config.  Two main possibilities are
considered below.  In all cases, it is necessary to encode, in a way or
another, that the config file uses the new syntax, so that Matplotlib can tell
which file-parser to use.

Maintain a matplotlibrc-like syntax
```````````````````````````````````

The config files would maintain the format

.. code:: conf

   key: value  # possible comment
   key: value

but all values would simply be parsed by passing to ``eval`` in the same
restricted environment as for cyclers.  Further validation of the inputs should
try to reuse whatever validation code Matplotlib already uses to validate
the same input when passed to an actual artist's property setter (e.g.,
validating a linestyle should call the same helper validator function as
``Line2D.set_linestyle``).

- The fact that a config file uses the nex syntax could be indicated by some
  "magic string" (e.g. ``# matplotlibrc-syntax-version: 2``), or a different
  naming convention.

- Parser handling for line-continuations would still need to be implemented.  A
  relatively simple possibility would be to support backslash continuations
  (lack of support for implicit continuations based on parentheses could be
  somewhat surprising to a user inputting Python syntax, though).

- From a security point of view, this is exactly as secure as the current
  situation (whatever one can pass to ``eval`` with this syntax, one could
  already do it by passing it as value for the ``axes.prop_cycle`` key).

- Support for ``patheffects`` would require adding more entries into the
  restricted environment.

Full Python syntax
``````````````````

The config files would simply be Python source files, of the form ::

   from matplotlib import rcParams
   rcParams["key"] = value  # possible comment
   rcParams["key"] = value

or ::

   from matplotlib import rcParams
   rcParams.update(
      {"key": value,  # possible comment
       "key": value}
   )

The files (with a ``.py`` extension, thus immediately distinguishable from
old-style configs) would be either

- option 1: ``exec``'d in a completely standard context (empty globals, all
  builtins available).  A few variables (``rcParams``, ``cycler``, etc.) could
  be preloaded into the globals, but I would prefer not (`#8235`_; see also
  `here <explicit-imports_>`_).

- option 2: Imported (operating by side-effect of the import), and then
  immediately removed from ``sys.modules`` so that reloading works; the config
  loader code would be in charge of locally patching ``sys.path`` to make the
  config files visible to the import system.

In either case, cascading style sheets can be implemented by having a config
file ``exec`` or ``import`` (depending on the option chosen) itself another
config file.

It would remain possible to disallow (accidental) modification of certain
rcParams from style files by locally patching ``RcParams.__setitem__`` in
``style.use``.  However, the style files would be able to execute arbitrary
code (this is a *feature* of this proposal).

As above, validation should share as much code as possible as the actual artist
property setters.

No parser would need to be written at all -- it's done for us by Python!

Direct loading from an URL would be disabled, as it is inherently insecure.
The documentation would encourage manual downloading (... or could even
document how to do it using ``urllib`` if we really want to) of style sheets,
which I believe is a good enough replacement (but I am happy to hear arguments
that it is not).

(Lack of) changes for the end-user
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the end user, calls to ``matplotlib.style.use`` are unchanged.  If
maintaining the *same* naming convention, then the lookup mechanism stays the
same and there is no ambiguity as to what file should be loaded.  If the naming
convention is changed (e.g. ``.mplstyle-v2`` with the matplotlibrc-like syntax,
or ``.py``\(!) with the full Python syntax), then a simple priority policy
(such as "always prefer the newer syntax if available") can be implemented.

Implementation
--------------

The general implementation strategy is outlined in the proposed solutions.
Neither strategy appears to present large technical difficulties.  Actual work
will be based on the agreed-upon syntax.

Backward compatibility
----------------------

New-style configs use a different code path, so old-style config support can
remain in order to maintain full backward compatibility.  Deprecating support
for old-style configs can be discussed and decided upon at a later time (or
never done).

Alternatives
------------

- Proposal:  Fix the current issues with the parsers and implement custom
  parsers for the additional kinds of values we want to support.

  Issues:  Is it really worth maintaining a large corpus of custom parsers for
  a custom-designed language that is essentially used only by us?

- Proposal:  Switch to another configuration language (JSON, YAML, etc.).

  Issues:  It remains necessary to be able to encode certain specific Python
  objects (certainly cyclers, possibly path effects), which means that they
  will need to be ``eval``'d (in which case I fail to see the advantage
  over using Python throughout), or that custom syntax (compatible with the
  underlying configuration language!) will need to be invented and custom
  parsers maintained.  Additionally, JSON does not support comments, and YAML
  is an extremely (overly, in my opinion) complex language.  See also the
  discussion that took place over PEP518_ (not that I particularly like the
  final choice of yet another obscure configuration language by that PEP).

.. _#3670: https://github.com/matplotlib/matplotlib/issues/3670
.. _#4240: https://github.com/matplotlib/matplotlib/issues/4240
.. _#4371: https://github.com/matplotlib/matplotlib/issues/4371
.. _#6157: https://github.com/matplotlib/matplotlib/issues/6157
.. _#6274: https://github.com/matplotlib/matplotlib/issues/6274
.. _#6320: https://github.com/matplotlib/matplotlib/issues/6320
.. _#7089: https://github.com/matplotlib/matplotlib/issues/7089
.. _#7219: https://github.com/matplotlib/matplotlib/issues/7219
.. _#8235: https://github.com/matplotlib/matplotlib/issues/8430
.. _#8430: https://github.com/matplotlib/matplotlib/issues/8430
.. _#9184: https://github.com/matplotlib/matplotlib/issues/9184
.. _PEP518: https://www.python.org/dev/peps/pep-0518/#other-file-formats
.. _explicit-imports: https://www.reddit.com/r/Python/comments/ex54j/seeking_clarification_on_pylonsturbogearspyramid/c1bo1v5/
