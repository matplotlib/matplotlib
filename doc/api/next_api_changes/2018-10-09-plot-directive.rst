Changes regarding the Sphinx plot directive
```````````````````````````````````````````

Fixed a bug in the Sphinx plot directive (.. plot:: path/to/plot_script.py)
where the source Python file was not being found relative to the directory of
the file containing the directive. In addition, its behavior was changed to
make it more streamlined with other Sphinx commands.

Documents that were using this feature may need to adjust the path argument
given to the plot directive. Two options are available:

1. Use absolute paths to find the file relative the ``plot_basedir`` (which
   defaults to the source directory, where conf.py is).
2. Use relative paths and the file is found relative to the directory of the
   file containing the directive.

Before this change, relative paths were resolved relative to ``plot_basedir``
(which defaulted to the source directory) and absolute paths were pointing to
files in the host system.

Since this will break documentations that were depending on the old behavior,
there is a deprecation period and a new configuration option is introduced to
get the old behavior. To get the old behavior specify
``plot_dir_resolve_method='absolute'`` in ``conf.py`` and specify
``plot_dir_resolve_method='relative'`` to get the new behavior. The old
behavior might be removed in future. The users are advised to switch to the new
behavior and fix the plot directives accordingly.
