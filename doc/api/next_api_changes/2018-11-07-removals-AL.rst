Removed rcParams
````````````````

The following deprecated rcParams are removed:

- ``text.dvipnghack``,
- ``nbagg.transparent`` (use :rc:`figure.facecolor` instead),
- ``plugins.directory``,
- ``axes.hold``,
- ``backend.qt4`` and ``backend.qt5`` (set the :envvar:`QT_API` environment
  variable instead).

The associated validator functions ``rcsetup.validate_qt4`` and
``validate_qt5`` are deprecated.
