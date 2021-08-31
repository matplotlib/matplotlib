import os
from pkg_resources import iter_entry_points
from .utils import trace


def iter_matching_entrypoints(path, entrypoint):
    trace("looking for ep", entrypoint, path)
    for ep in iter_entry_points(entrypoint):
        if os.path.exists(os.path.join(path, ep.name)):
            if os.path.isabs(ep.name):
                trace("ignoring bad ep", ep)
            trace("found ep", ep)
            yield ep
