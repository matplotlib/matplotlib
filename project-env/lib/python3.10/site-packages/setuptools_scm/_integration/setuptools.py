from __future__ import annotations

import os
from typing import IO


def read_dist_name_from_setup_cfg(
    input: str | os.PathLike[str] | IO[str] = "setup.cfg",
) -> str | None:

    # minimal effort to read dist_name off setup.cfg metadata
    import configparser

    parser = configparser.ConfigParser()

    if isinstance(input, (os.PathLike, str)):
        parser.read([input])
    else:
        parser.read_file(input)

    dist_name = parser.get("metadata", "name", fallback=None)
    return dist_name
