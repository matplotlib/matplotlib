from .backend_qt5cairo import _BackendQT5Cairo


@_BackendQT5Cairo.export
class _BackendQT4Cairo(_BackendQT5Cairo):
    required_interactive_framework = "qt4"
