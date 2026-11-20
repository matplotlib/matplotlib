from typing import Any

QT_API_PYQT6: str
QT_API_PYSIDE6: str
QT_API_PYQT5: str
QT_API_ENV: str | None
QT_API: str | None

# The concrete Qt binding is chosen at runtime and has no type stubs, so the
# re-exported submodules and helpers are typed as ``Any``.
QtCore: Any
QtGui: Any
QtWidgets: Any
QtSvg: Any
__version__: str

def _setup_pyqt5plus() -> None: ...
def _isdeleted(obj: Any) -> bool: ...
def _to_int(x: Any) -> int: ...
def _exec(obj: Any) -> None: ...
