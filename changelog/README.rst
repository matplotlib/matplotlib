:orphan:

Changelog
=========

This directory contains "news fragments" which are short files that contain a
small **ReST**-formatted text that will be added to the next what's new page.

Make sure to use full sentences with correct case and punctuation, and please
try to use Sphinx intersphinx using backticks.

Each file should be named like ``<PULL REQUEST>.<TYPE>.rst``, where
``<PULL REQUEST>`` is a pull request number, and ``<TYPE>`` is one of:

* ``breaking``: A change which requires users to change code and is not
  backwards compatible. (Not to be used for removal of deprecated features.)
* ``feature``: New user facing features and any new behavior.
* ``removal``: Removal of a deprecated part of the API.
* ``bugfix``: Any bug fixes that do not require users to change code.

So for example: ``123.feature.rst``, ``456.api_change.rst``.

If you are unsure what pull request type to use, don't hesitate to ask in your
PR.

Note that the ``towncrier`` tool will automatically reflow your text, so it
will work best if you stick to a single paragraph, but multiple sentences and
links are OK and encouraged.

You can install ``towncrier`` and then run ``towncrier --draft`` if you want to
get a preview of how your change will look in the final release notes.

.. note::

    This README was adapted from the pytest changelog readme under the terms of
    the MIT licence.
