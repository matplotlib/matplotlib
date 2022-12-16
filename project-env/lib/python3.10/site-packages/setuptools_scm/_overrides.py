from __future__ import annotations

import os

from .config import Configuration
from .utils import trace
from .version import meta
from .version import ScmVersion


PRETEND_KEY = "SETUPTOOLS_SCM_PRETEND_VERSION"
PRETEND_KEY_NAMED = PRETEND_KEY + "_FOR_{name}"


def _read_pretended_version_for(config: Configuration) -> ScmVersion | None:
    """read a a overridden version from the environment

    tries ``SETUPTOOLS_SCM_PRETEND_VERSION``
    and ``SETUPTOOLS_SCM_PRETEND_VERSION_FOR_$UPPERCASE_DIST_NAME``
    """
    trace("dist name:", config.dist_name)
    pretended: str | None
    if config.dist_name is not None:
        pretended = os.environ.get(
            PRETEND_KEY_NAMED.format(name=config.dist_name.upper())
        )
    else:
        pretended = None

    if pretended is None:
        pretended = os.environ.get(PRETEND_KEY)

    if pretended:
        # we use meta here since the pretended version
        # must adhere to the pep to begin with
        return meta(tag=pretended, preformatted=True, config=config)
    else:
        return None
