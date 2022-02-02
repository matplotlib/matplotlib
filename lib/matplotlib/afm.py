from matplotlib._afm import *  # noqa: F401, F403
from matplotlib import _api
_api.warn_deprecated(
    "3.6", message="The module %(name)s is deprecated since %(since)s.",
    name=f"{__name__}")
