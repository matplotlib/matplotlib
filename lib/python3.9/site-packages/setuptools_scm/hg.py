import os
from .config import Configuration
from .utils import do, trace, data_from_mime, require_command
from .version import meta, tags_to_versions


def _hg_tagdist_normalize_tagcommit(config, tag, dist, node, branch):
    dirty = node.endswith("+")
    node = "h" + node.strip("+")

    # Detect changes since the specified tag
    revset = (
        "(branch(.)"  # look for revisions in this branch only
        " and tag({tag!r})::."  # after the last tag
        # ignore commits that only modify .hgtags and nothing else:
        " and (merge() or file('re:^(?!\\.hgtags).*$'))"
        " and not tag({tag!r}))"  # ignore the tagged commit itself
    ).format(tag=tag)
    if tag != "0.0":
        commits = do(
            ["hg", "log", "-r", revset, "--template", "{node|short}"],
            config.absolute_root,
        )
    else:
        commits = True
    trace("normalize", locals())
    if commits or dirty:
        return meta(
            tag, distance=dist, node=node, dirty=dirty, branch=branch, config=config
        )
    else:
        return meta(tag, config=config)


def parse(root, config=None):
    if not config:
        config = Configuration(root=root)

    require_command("hg")
    identity_data = do("hg id -i -b -t", config.absolute_root).split()
    if not identity_data:
        return
    node = identity_data.pop(0)
    branch = identity_data.pop(0)
    if "tip" in identity_data:
        # tip is not a real tag
        identity_data.remove("tip")
    tags = tags_to_versions(identity_data)
    dirty = node[-1] == "+"
    if tags:
        return meta(tags[0], dirty=dirty, branch=branch, config=config)

    if node.strip("+") == "0" * 12:
        trace("initial node", config.absolute_root)
        return meta("0.0", config=config, dirty=dirty, branch=branch)

    try:
        tag = get_latest_normalizable_tag(config.absolute_root)
        dist = get_graph_distance(config.absolute_root, tag)
        if tag == "null":
            tag = "0.0"
            dist = int(dist) + 1
        return _hg_tagdist_normalize_tagcommit(config, tag, dist, node, branch)
    except ValueError:
        pass  # unpacking failed, old hg


def get_latest_normalizable_tag(root):
    # Gets all tags containing a '.' (see #229) from oldest to newest
    cmd = [
        "hg",
        "log",
        "-r",
        "ancestors(.) and tag('re:\\.')",
        "--template",
        "{tags}\n",
    ]
    outlines = do(cmd, root).split()
    if not outlines:
        return "null"
    tag = outlines[-1].split()[-1]
    return tag


def get_graph_distance(root, rev1, rev2="."):
    cmd = ["hg", "log", "-q", "-r", f"{rev1}::{rev2}"]
    out = do(cmd, root)
    return len(out.strip().splitlines()) - 1


def archival_to_version(data, config=None):
    trace("data", data)
    node = data.get("node", "")[:12]
    if node:
        node = "h" + node
    if "tag" in data:
        return meta(data["tag"], config=config)
    elif "latesttag" in data:
        return meta(
            data["latesttag"],
            distance=data["latesttagdistance"],
            node=node,
            config=config,
        )
    else:
        return meta("0.0", node=node, config=config)


def parse_archival(root, config=None):
    archival = os.path.join(root, ".hg_archival.txt")
    data = data_from_mime(archival)
    return archival_to_version(data, config=config)
