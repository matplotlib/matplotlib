from __future__ import annotations

from logging import getLogger

from packaging.version import InvalidVersion
from packaging.version import Version as Version


class NonNormalizedVersion(Version):
    """A non-normalizing version handler.

    You can use this class to preserve version verification but skip normalization.
    For example you can use this to avoid git release candidate version tags
    ("1.0.0-rc1") to be normalized to "1.0.0rc1". Only use this if you fully
    trust the version tags.
    """

    def __init__(self, version: str) -> None:
        # parse and validate using parent
        super().__init__(version)

        # store raw for str
        self._raw_version = version

    def __str__(self) -> str:
        # return the non-normalized version (parent returns the normalized)
        return self._raw_version

    def __repr__(self) -> str:
        # same pattern as parent
        return f"<NonNormalizedVersion({self._raw_version!r})>"


def _version_as_tuple(version_str: str) -> tuple[int | str, ...]:
    try:
        parsed_version = Version(version_str)
    except InvalidVersion:

        log = getLogger("setuptools_scm")
        log.exception("failed to parse version %s", version_str)
        return (version_str,)
    else:
        version_fields: tuple[int | str, ...] = parsed_version.release
        if parsed_version.dev is not None:
            version_fields += (f"dev{parsed_version.dev}",)
        if parsed_version.local is not None:
            version_fields += (parsed_version.local,)
        return version_fields
