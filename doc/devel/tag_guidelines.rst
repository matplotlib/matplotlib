Tagging guidelines
==================

Why do we need tags?
--------------------

Tags serve multiple purposes.

Tags have a one-to-many organization (i.e. one example can have several tags), while
the gallery structure requires that examples are placed in only one location. This means
tags provide a secondary layer of organization and make the gallery of examples more
flexible and more user-friendly.

They allow for better discoverability, search, and browse functionality. They are
helpful for users struggling to write a search query for what they're looking for.

Hidden tags provide additional functionality for maintainers and contributors.

How to tag?
-----------
Place the tag directive at the bottom of each page and add the tags underneath, e.g.:

.. code-block:: rst

    .. tags::
       topic: tagging, purpose: reference

What gets a tag?
----------------

Every gallery example should be tagged with:

* 1+ content tags
* structural, domain, or internal tag(s) if helpful

Tags can repeat existing forms of organization (e.g. an example is in the Animation
folder and also gets an ``animation`` tag).

Tags are helpful to denote particularly good "byproduct" examples. E.g. the explicit
purpose of a gallery example might be to demonstrate a colormap, but it's also a good
demonstration of a legend. Tag ``legend`` to indicate that, rather than changing the
title or the scope of the example.

.. card::

    **Tag Categories**
    ^^^
    .. rst-class:: section-toc

    .. toctree::
        :maxdepth: 2

        tag_glossary

    +++
    See :doc:`Tag Glossary <tag_glossary>` for a complete list

Proposing new tags
------------------

1. Review existing tag list, looking out for similar entries (i.e. ``axes`` and ``axis``).
2. If a relevant tag or subcategory does not yet exist, propose it. Each tag is two
   parts: ``subcategory: tag``. Tags should be one or two words.
3. New tags should be added when they are relevant to existing gallery entries too.
   Avoid tags that will link to only a single gallery entry.
4. Tags can recreate other forms of organization.

Tagging organization aims to work for 80-90% of cases. Some examples fall outside of the
tagging structure. Niche or specific examples shouldn't be given standalone tags that
won't apply to other examples.
