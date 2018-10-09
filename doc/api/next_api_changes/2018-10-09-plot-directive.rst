Changes regarding the Sphinx plot directive
```````````````````````````````````````````

Fixed a bug in the Sphinx plot directive (.. plot:: path/to/plot_script.py)
where the source Python file was not being found relative to the directory of
the file containing the directive. In addition, its behavior was changed to
make it more streamlined with other Sphinx commands.

Documents that were using this feature may need to adjust the path argument
given to the plot directive. Two options are available:

1. Use absolute paths to find the file relative the ``plot_basedir`` (which
   defaults to the directory where conf.py is).
2. Use relative paths and the file is found relative to the directory of the
   file containing the directive.

Before this change, relative paths were resolved relative to the source
directory (where conf.py is) and absolute paths were pointing to files in the
host system.
