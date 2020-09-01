from matplotlib import cbook
from ._formsubplottool import UiSubplotTool


cbook.warn_deprecated(
    "3.3", obj_type="module", name=__name__,
    alternative="matplotlib.backends.backend_qt5.SubplotToolQt")
__all__ = ["UiSubplotTool"]
