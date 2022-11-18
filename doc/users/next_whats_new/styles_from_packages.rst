Style files can be imported from third-party packages
-----------------------------------------------------

Third-party packages can now distribute style files that are globally available
as follows.  Assume that a package is importable as ``import mypackage``, with
a ``mypackage/__init__.py`` module.  Then a ``mypackage/presentation.mplstyle``
style sheet can be used as ``plt.style.use("mypackage.presentation")``.

The implementation does not actually import ``mypackage``, making this process
safe against possible import-time side effects.  Subpackages (e.g.
``dotted.package.name``) are also supported.
