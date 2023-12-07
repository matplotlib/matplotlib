import logging

from pyvirtualdisplay.about import __version__
from pyvirtualdisplay.display import Display

Display  # ignore unused

log = logging.getLogger(__name__)

log = logging.getLogger(__name__)
log.debug("version=%s", __version__)
