:orphan:

Adding API change notes
=======================

API change notes for future releases are collected in the most recent directory
:file:`api_changes_X.Y`. They are divided into four categories:

- **Deprecations**: Announcements of future changes. Typically, these will
  raise a deprecation warning and users of this API should change their code
  to stay compatible with future releases of Matplotlib. If possible, state
  what should be used instead.
- **Removals**: Parts of the API that got removed. If possible, state what
  should be used instead.
- **Behaviour changes**: API that stays valid but will yield a different
  result.
- **Development changes**: Changes to the build process, dependencies, etc.

Please place new entries in the respective files in this directory. Typically,
each change will get its own section, but you may also amend existing sections
when suitable. The overall goal is a comprehensible documentation of the
changes.

A typical entry could look like this::

    Locators
    ~~~~~~~~
    The unused `Locator.autoscale()` method is deprecated (pass the axis
    limits to `Locator.view_limits()` instead).
