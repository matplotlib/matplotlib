import os
from .utils import data_from_mime, trace
from .version import tag_to_version, meta


def parse_pkginfo(root, config=None):

    pkginfo = os.path.join(root, "PKG-INFO")
    trace("pkginfo", pkginfo)
    data = data_from_mime(pkginfo)
    version = data.get("Version")
    if version != "UNKNOWN":
        return meta(version, preformatted=True, config=config)


def parse_pip_egg_info(root, config=None):
    pipdir = os.path.join(root, "pip-egg-info")
    if not os.path.isdir(pipdir):
        return
    items = os.listdir(pipdir)
    trace("pip-egg-info", pipdir, items)
    if not items:
        return
    return parse_pkginfo(os.path.join(pipdir, items[0]), config=config)


def fallback_version(root, config=None):
    if config.parentdir_prefix_version is not None:
        _, parent_name = os.path.split(os.path.abspath(root))
        if parent_name.startswith(config.parentdir_prefix_version):
            version = tag_to_version(
                parent_name[len(config.parentdir_prefix_version) :], config
            )
            if version is not None:
                return meta(str(version), preformatted=True, config=config)
    if config.fallback_version is not None:
        return meta(config.fallback_version, preformatted=True, config=config)
