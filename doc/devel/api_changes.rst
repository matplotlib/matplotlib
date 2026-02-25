.. _api_changes:

API guidelines
==============

API consistency and stability are of great value; Therefore, API changes
(e.g. signature changes, behavior changes, removals) will only be conducted
if the added benefit is worth the effort of adapting existing code.

Because we are a visualization library, our primary output is the final
visualization the user sees; therefore, the appearance of the figure is part of
the API and any changes, either semantic or aesthetic, are backwards-incompatible
API changes.


Add new API
-----------

Every new function, parameter and attribute that is not explicitly marked as
private (i.e., starts with an underscore) becomes part of Matplotlib's public
API. As discussed above, changing the existing API is cumbersome. Therefore,
take particular care when adding new API:

- Mark helper functions and internal attributes as private by prefixing them
  with an underscore.
- Carefully think about good names for your functions and variables.
- Try to adopt patterns and naming conventions from existing parts of the
  Matplotlib API.
- Consider making as many arguments keyword-only as possible. See also
  `API Evolution the Right Way -- Add Parameters Compatibly`__.

  __ https://emptysqua.re/blog/api-evolution-the-right-way/#adding-parameters


Add new rcParams
^^^^^^^^^^^^^^^^

When adding a new rcParam, the following files must be updated:

1. :file:`lib/matplotlib/rcsetup.py` - Add a validator entry to the
   ``_validators`` dict and a corresponding ``_Param`` entry with default value
   and description.
2. :file:`lib/matplotlib/mpl-data/matplotlibrc` - Add a commented-out entry
   showing the default value.
3. :file:`lib/matplotlib/typing.py` - Add the key to the ``RcKeyType`` Literal
   so that it is recognized as a valid rcParam key.


Add or change colormaps, color sequences, and styles
----------------------------------------------------
Visual changes are considered an API break. Therefore, we generally do not modify
existing colormaps, color sequences, or styles.

We put a high bar on adding new colormaps and styles to prevent excessively growing
them. While the decision is case-by-case, evaluation criteria include:

- novelty: Does it support a new use case? e.g. slight variations of existing maps,
  sequences and styles are likely not accepted.
- usability and accessibility: Are colors of sequences sufficiently distinct? Has
  colorblindness been considered?
- evidence of wide spread usage: for example academic papers, industry blogs and
  whitepapers, or inclusion in other visualization libraries or domain specific tools
- open license: colormaps, sequences, and styles must have a BSD compatible license
  (see :ref:`license-discussion`)

.. _deprecation-guidelines:

Deprecate API
-------------

When deciding to deprecate API we carefully consider the balance between the advantages
(clearer interfaces, better usability, less maintenance) and the disadvantages (users
have to learn new API and have to modify existing code).

.. tip::

   A rough estimate on the current usage of an API can be obtained by a GitHub code
   search. A good search pattern is typically
   ``[expression] language:Python NOT is:fork``. ``[expression]`` may be a simple
   string, but often regular expressions are helpful to exclude incorrect matches.
   You can start simple and look at the search results, if there are too many
   incorrect matches, gradually refine your search criteria.

   It can also be helpful to add ``NOT path:**/matplotlib/** NOT path:**/site-packages/**``
   to exclude matches where the matplotlib codebase is checked into another repo,
   either as direct sources or as part of an environment.

   *Example*: Calls of the method ``Figure.draw()`` could be matched using
   ``/\bfig(ure)?\.draw\(/``. This expression employs a number of patterns:

   - Add the opening bracket ``(`` after the method name to only find method calls.
   - Include a common object name if there are otherwise too many false positives.
     There are many ``draw()`` functions out there, but the ones we are looking for
     are likely called via ``fig.draw()`` or ``figure.draw()``.
   - Use the word boundary marker ``\b`` to make sure your expression is not a
     matching as part of a longer word.

   `Link to the resulting GitHub search <https://github.com/search?q=%2F%5Cbfig%28ure%29%3F%5C.draw%5C%28%2F+language%3APython+NOT+is%3Afork&type=code>`_


API changes in Matplotlib have to be performed following the deprecation process
below, except in very rare circumstances as deemed necessary by the development
team. Generally API deprecation happens in two stages:

* **introduce:** warn users that the API *will* change
* **expire:** API *is* changed as described in the introduction period

This ensures that users are notified before the change will take effect and thus
prevents unexpected breaking of code. Occasionally deprecations are marked as
**pending**, which means that the deprecation will be introduced in a future release.

Rules
^^^^^
- Deprecations are targeted at the next :ref:`meso release <pr-milestones>` (e.g. 3.Y)
- Deprecated API is generally removed (expired) two point-releases after introduction
  of the deprecation. Longer deprecations can be imposed by core developers on
  a case-by-case basis to give more time for the transition
- The old API must remain fully functional during the deprecation period
- If alternatives to the deprecated API exist, they should be available
  during the deprecation period
- If in doubt, decisions about API changes are finally made by the
  `API consistency lead <https://matplotlib.org/governance/people.html>`_ developer.


.. _intro-deprecation:

Introduce deprecation
^^^^^^^^^^^^^^^^^^^^^

Deprecations are introduced to warn users that the API will change. The deprecation
notice describes how the API will change. When alternatives to the deprecated API exist,
they are also listed in the notice and decorators.

#. Create a :ref:`deprecation notice <api_whats_new>`

#. If possible, issue a `~matplotlib.MatplotlibDeprecationWarning` when the
   deprecated API is used. There are a number of helper tools for this:

   - Use ``_api.warn_deprecated()`` for general deprecation warnings
   - Use the decorator ``@_api.deprecated`` to deprecate classes, functions,
     methods, or properties
   - Use ``@_api.deprecate_privatize_attribute`` to annotate deprecation of
     attributes while keeping the internal private version.
   - To warn on changes of the function signature, use the decorators
     ``@_api.delete_parameter``, ``@_api.rename_parameter``, and
     ``@_api.make_keyword_only``

   All these helpers take a first parameter *since*, which should be set to
   the next point release, e.g. "3.x".

   You can use standard rst cross references in *alternative*.

#. Make appropriate changes to the type hints in the associated ``.pyi`` file.
   The general guideline is to match runtime reported behavior.

   - Items marked with ``@_api.deprecated`` or ``@_api.deprecate_privatize_attribute``
     are generally kept during the expiry period, and thus no changes are needed on
     introduction.
   - Items decorated with ``@_api.rename_parameter`` or ``@_api.make_keyword_only``
     report the *new* (post deprecation) signature at runtime, and thus *should* be
     updated on introduction.
   - Items decorated with ``@_api.delete_parameter`` should include a default value hint
     for the deleted parameter, even if it did not previously have one (e.g.
     ``param: <type> = ...``).

.. _expire-deprecation:

Expire deprecation
^^^^^^^^^^^^^^^^^^
The API changes described in the introduction notice are only implemented after the
introduction period has expired.

#. Create a :ref:`deprecation announcement <api_whats_new>`. For the content,
   you can usually copy the deprecation notice and adapt it slightly.

#. Change the code functionality and remove any related deprecation warnings.

#. Make appropriate changes to the type hints in the associated ``.pyi`` file.

   - Items marked with ``@_api.deprecated`` or ``@_api.deprecate_privatize_attribute``
     are to be removed on expiry.
   - Items decorated with ``@_api.rename_parameter`` or ``@_api.make_keyword_only``
     will have been updated at introduction, and require no change now.
   - Items decorated with ``@_api.delete_parameter`` will need to be updated to the
     final signature, in the same way as the ``.py`` file signature is updated.
   - Any entries in :file:`ci/mypy-stubtest-allowlist.txt` which indicate a deprecation
     version should be double checked. In most cases this is not needed, though some
     items were never type hinted in the first place and were added to this file
     instead. For removed items that were not in the stub file, only deleting from the
     allowlist is required.

.. _pending-deprecation:

Pending deprecation
^^^^^^^^^^^^^^^^^^^

A pending deprecation is an announcement that a deprecation will be introduced in the
future. By default, pending deprecations do not raise a warning to the user; however,
pending deprecations are rendered in the documentation and listed in the release notes.
Pending notices are primarily intended to give downstream library and tool developers
time to adapt their code so that it does not raise a deprecation
warning. This is because their users cannot act on warnings triggered by how the tools
and libraries use Matplotlib. It's also possible to run Python in dev mode to raise
`PendingDeprecationWarning`.

To mark a deprecation as pending, set the following parameters on the appropriate
deprecation decorator:
* the *pending* parameter is set to ``True``
* the *removal* parameter is left blank

When converting a pending deprecation to an introduced deprecation, update the
decorator such that:
* *pending* is set to ``False``
* *since* is set to the next meso release (3.Y+1)
* *removal* is set to at least 2 meso releases after (3.Y+3) introduction.

Pending deprecations are documented in the :ref:`API change notes <api_whats_new>` in
the same manner as introduced and expired deprecations. The notice should include
*pending deprecation* in the title.


.. redirect-from:: /devel/coding_guide#new-features-and-api-changes

.. _api_whats_new:

Announce new and deprecated API
-------------------------------

When adding or changing the API in a backward in-compatible way, please add the
appropriate :ref:`versioning directive <versioning-directives>` and document it
in the :ref:`release notes <release-notes>` by adding an entry to the appropriate
folder:

+-------------------+-----------------------------+----------------------------------------------+
|                   |   versioning directive      |  announcement folder                         |
+===================+=============================+==============================================+
| new feature       | ``.. versionadded:: 3.N``   | :file:`doc/release/next_whats_new/`          |
+-------------------+-----------------------------+----------------------------------------------+
| API change        | ``.. versionchanged:: 3.N`` | :file:`doc/api/next_api_changes/[kind]`      |
+-------------------+-----------------------------+----------------------------------------------+

When deprecating API, please add a notice as described in the
:ref:`deprecation guidelines <deprecation-guidelines>` and summarized here:

+--------------------------------------------------+----------------------------------------------+
|   stage                                          |             announcement folder              |
+===========+======================================+==============================================+
| :ref:`introduce deprecation <intro-deprecation>` | :file:`doc/api/next_api_changes/deprecation` |
+-----------+--------------------------------------+----------------------------------------------+
| :ref:`expire deprecation <expire-deprecation>`   | :file:`doc/api/next_api_changes/[kind]`      |
+-----------+--------------------------------------+----------------------------------------------+

Generally the introduction notices can be repurposed for the expiration notice as they
are expected to be describing the same API changes and removals.

.. _versioning-directives:

Versioning directives
^^^^^^^^^^^^^^^^^^^^^

When making a backward incompatible change, please add a versioning directive in
the docstring. The directives should be placed at the end of a description block.
For example::

  class Foo:
      """
      This is the summary.

      Followed by a longer description block.

      Consisting of multiple lines and paragraphs.

      .. versionadded:: 3.5

      Parameters
      ----------
      a : int
          The first parameter.
      b: bool, default: False
          This was added later.

          .. versionadded:: 3.6
      """

      def set_b(b):
          """
          Set b.

          .. versionadded:: 3.6

          Parameters
          ----------
          b: bool

For classes and functions, the directive should be placed before the
*Parameters* section. For parameters, the directive should be placed at the
end of the parameter description. The micro release version is omitted and
the directive should not be added to entire modules.

.. _release-notes:

Release notes
^^^^^^^^^^^^^

For both change notes and what's new, please avoid using cross-references in section
titles as it causes links to be confusing in the table of contents. Instead, ensure that
a cross-reference is included in the descriptive text.

.. _api-change-notes:

API change notes
""""""""""""""""

.. include:: ../api/next_api_changes/README.rst
   :start-after: api-change-guide-start
   :end-before: api-change-guide-end

.. _whats-new-notes:

What's new notes
""""""""""""""""

.. include:: ../release/next_whats_new/README.rst
   :start-after: whats-new-guide-start
   :end-before: whats-new-guide-end

Discourage API
--------------

We have API that we do not recommend anymore for new code, but that cannot be
deprecated because its removal would be breaking backward-compatibility and too
disruptive. In such a case we can formally discourage API. This can cover
specific parameters, call patterns, whole methods etc.

To do so, add a note to the docstring ::

    .. admonition:: Discouraged

       [description and suggested alternative]

You find several examples for good descriptions if you search the codebase for
``.. admonition:: Discouraged``.

Additionally, if a whole function is discouraged, prefix the summary line with
``[*Discouraged*]`` so that it renders in the API overview like this

    [*Discouraged*] Return the XAxis instance.
