=======================
 MEP26: Artist styling
=======================

.. contents::
   :local:


Status
======

**Rejected**

Branches and Pull requests
==========================

Abstract
========

This MEP proposes a new stylesheet implementation to allow more
comprehensive and dynamic styling of artists.

The current version of matplotlib (1.4.0) allows stylesheets based on
the rcParams syntax to be applied before creation of a plot.  The
methodology below proposes a new syntax, based on CSS, which would
allow styling of individual artists and properties, which can be
applied dynamically to existing objects.

This is related to (and makes steps toward) the overall goal of moving
to a DOM/tree-like architecture.


Detailed description
====================

Currently, the look and appearance of existing artist objects (figure,
axes, Line2D etc...) can only be updated via `set_` and `get_` methods
on the artist object, which is quite laborious, especially if no
reference to the artist(s) has been stored.  The new style sheets
introduced in 1.4 allow styling before a plot is created, but do not
offer any means to dynamically update plots or distinguish between
artists of the same type (i.e. to specify the `line color` and `line
style` separately for differing `Line2D` objects).

The initial development should concentrate on allowing styling of
artist primitives (those `artists` that do not contain other
`artists`), and further development could expand the CSS syntax rules
and parser to allow more complex styling. See the appendix for a list
of primitives.

The new methodology would require development of a number of steps:

- A new stylesheet syntax (likely based on CSS) to allow selection of
  artists by type, class, id etc...
- A mechanism by which to parse a stylesheet into a tree
- A mechanism by which to translate the parse-tree into something
  which can be used to update the properties of relevant
  artists. Ideally this would implement a method by which to traverse
  the artists in a tree-like structure.
- A mechanism by which to generate a stylesheet from existing artist
  properties. This would be useful to allow a user to export a
  stylesheet from an existing figure (where the appearance may have
  been set using the matplotlib API)...

Implementation
==============

It will be easiest to allow a '3rd party' to modify/set the style of
an artist if the 'style' is created as a separate class and store
against the artist as a property.  The `GraphicsContext` class already
provides a the basis of a `Style` class and an artists `draw` method can
be refactored to use the `Style` class rather than setting up it's own
`GraphicsContext` and transferring it's style-related properties to
it.  A minimal example of how this could be implemented is shown here:
https://github.com/JamesRamm/mpl_experiment

IMO, this will also make the API and code base much neater as
individual get/set methods for artist style properties are now
redundant...  Indirectly related would be a general drive to replace
get/set methods with properties. Implementing the style class with
properties would be a big stride toward this...

For initial development, I suggest developing a syntax based on a much
(much much) simplified version of CSS. I am in favour of dubbing this
Artist Style Sheets :+1: :

BNF Grammar
-----------

I propose a very simple syntax to implement initially (like a proof of
concept), which can be expanded upon in the future. The BNF form of
the syntax is given below and then explained ::

    RuleSet ::= SelectorSequence "{"Declaration"}"

    SelectorSequence :: = Selector {"," Selector}

    Declaration ::= propName":" propValue";"

    Selector ::= ArtistIdent{"#"Ident}

    propName ::= Ident

    propValue ::= Ident | Number | Colour | "None"

`ArtistIdent`, `Ident`, `Number` and `Colour` are tokens (the basic
building blocks of the expression) which are defined by regular
expressions.

Syntax
------

A CSS stylesheet consists of a series of **rule sets** in hierarchical
order (rules are applied from top to bottom). Each rule follows the
syntax ::

    selector {attribute: value;}

Each rule can have any number of `attribute`: `value` pairs, and a
stylesheet can have any number of rules.

The initial syntax is designed only for `artist` primitives. It does
not address the question of how to set properties on `container` types
(whose properties may themselves be `artists` with settable
properties), however, a future solution to this could simply be nested
`RuleSet` s

Selectors
~~~~~~~~~


Selectors define the object to which the attribute updates should be
applied. As a starting point, I propose just 2 selectors to use in
initial development:



Artist Type Selector


Select an `artist` by it's type. E.g `Line2D` or `Text`::

    Line2D {attribute: value}

The regex for matching the artist type selector (`ArtistIdent` in the BNF grammar) would be::

    ArtistIdent = r'(?P<ArtistIdent>\bLine2D\b|\bText\b|\bAxesImage\b|\bFigureImage\b|\bPatch\b)'

GID selector
~~~~~~~~~~~~

Select an `artist` by its `gid`::

    Line2D#myGID {attribute: value}

A `gid` can be any string, so the regex could be as follows::

    Ident = r'(?P<Ident>[a-zA-Z_][a-zA-Z_0-9]*)'


The above selectors roughly correspond to their CSS counterparts
(http://www.w3.org/TR/CSS21/selector.html)

Attributes and values
~~~~~~~~~~~~~~~~~~~~~

- `Attributes` are any valid (settable) property for the `artist` in question.
- `Values` are any valid value for the property (Usually a string, or number).

Parsing
-------

Parsing would consist of breaking the stylesheet into tokens (the
python cookbook gives a nice tokenizing recipe on page 66), applying
the syntax rules and constructing a `Tree`. This requires defining the
grammar of the stylesheet (again, we can borrow from CSS) and writing
a parser. Happily, there is a recipe for this in the python cookbook
aswell.


Visitor pattern for matplotlib figure
-------------------------------------

In order to apply the stylesheet rules to the relevant artists, we
need to 'visit' each artist in a figure and apply the relevant rule.
Here is a visitor class (again, thanks to python cookbook), where each
`node` would be an artist in the figure. A `visit_` method would need
to be implemented for each mpl artist, to handle the different
properties for each ::

    class Visitor:
        def visit(self, node):
           name = 'visit_' + type(node).__name__
           meth = getattr(self, name, None)
           if meth is None:
              raise NotImplementedError
           return meth(node)

An `evaluator` class would then take the stylesheet rules and
implement the visitor on each one of them.



Backward compatibility
======================

Implementing a separate `Style` class would break backward
compatibility as many get/set methods on an artist would become
redundant.  While it would be possible to alter these methods to hook
into the `Style` class (stored as a property against the artist), I
would be in favor of simply removing them to both neaten/simplify the
codebase and to provide a simple, uncluttered API...

Alternatives
============

No alternatives, but some of the ground covered here overlaps with
MEP25, which may assist in this development

Appendix
========

Matplotlib primitives
---------------------

This will form the initial selectors which stylesheets can use.

* Line2D
* Text
* AxesImage
* FigureImage
* Patch
