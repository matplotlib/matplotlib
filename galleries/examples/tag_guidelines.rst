Guidelines for assigning tags to gallery examples
=================================================

Why do we need tags?
--------------------

Tags serve multiple purposes.

Tags have a one-to-many organization (i.e. one example can have several tags), while the gallery structure requires that examples are placed in only one location. This means tags provide a secondary layer of organization and make the gallery of examples more flexible and more user-friendly.

They allow for better discoverability, search, and browse functions. They are helpful for users struggling to write a search query for what they're looking for.

Hidden tags provide additional functionality for maintainers and contributors.

What gets a tag?
----------------

Every gallery example should be tagged with:

* 1+ content tags
* structural, domain, or internal tag(s) if helpful

Examples with many tags may indicate that the entry is a showcase example.

Tags can repeat existing forms of organization (e.g. an example is in the Animation folder and also gets an ``animation`` tag).

Tags are helpful to denote particularly good "byproduct" examples. E.g. the explicit purpose of a gallery example might be to demonstrate a colormap, but it's also a good demonstration of a legend. Tag ``legend`` to indicate that, rather than changing the title or the scope of the example.

**Tag Categories** - See :doc:`Tag Glossary <tag_glossary>` for a complete list of tags.

I. API tags: what content from the API reference is in the example?
II. Structural tags: what format is the example? What context can we provide?
III. Domain tags: what discipline(s) might seek this example consistently?
IV. Internal tags: what information is helpful for maintainers or contributors?

Proposing new tags
------------------

1. Review existing tag list, looking out for similar entries (i.e. ``axes`` and ``axis``).
2. If a relevant tag or subcategory does not yet exist, propose it. Each tag is two parts: ``subcategory: tag``. Tags should be one or two words.
3. New tags should be be added when they are relevant to existing gallery entries too. Avoid tags that will link to only a single gallery entry.
4. Tags can recreate other forms of organization.

Note: Tagging organization aims to work for 80-90% of cases. Some examples fall outside of the tagging structure. Niche or specific examples shouldn't be given standalone tags that won't apply to other examples.

How to tag?
-----------
 where you put the directive is where the tag will show

Related content
---------------

What is a gallery example?
^^^^^^^^^^^^^^^^^^^^^^^^^^

The gallery of examples contains visual demonstrations of matplolib features. Gallery examples exist so that users can scan through visual examples.

Unlike tutorials or user guides, gallery examples teach by demonstration, rather than by explanation or instruction.

Gallery examples should avoid instruction or excessive explanation except for brief clarifying code comments. Instead, they can tag related concepts and/or link to relevant tutorials or user guides.

Format
^^^^^^

All gallery examples should aim to follow the following format:

* Title: 1-6 words, descriptive of content
* Subtitle: 10-50 words, action-oriented description of the example subject
* Image: a clear demonstration of the subject, showing edge cases and different applications if possible
* Code + Text (optional): code, commented as appropriate + written text to add context if necessary

Example:

``bbox_intersect`` gallery example showcases the point of visual examples

* messy example/ hard to categorize, but the gallery is definitely the right spot for it because it's the kind of thing you'd find by visual search
* https://matplotlib.org/devdocs/gallery/misc/bbox_intersect.html#sphx-glr-gallery-misc-bbox-intersect-py
