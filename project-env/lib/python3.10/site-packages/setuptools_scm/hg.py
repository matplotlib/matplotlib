from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import TYPE_CHECKING

from ._version_cls import Version
from .config import Configuration
from .scm_workdir import Workdir
from .utils import data_from_mime
from .utils import do_ex
from .utils import require_command
from .utils import trace
from .version import meta
from .version import ScmVersion
from .version import tag_to_version

if TYPE_CHECKING:
    from . import _types as _t


class HgWorkdir(Workdir):

    COMMAND = "hg"

    @classmethod
    def from_potential_worktree(cls, wd: _t.PathT) -> HgWorkdir | None:
        require_command(cls.COMMAND)
        root, err, ret = do_ex("hg root", wd)
        if ret:
            return None
        return cls(root)

    def get_meta(self, config: Configuration) -> ScmVersion | None:

        node: str
        tags_str: str
        bookmark: str
        node_date_str: str
        node, tags_str, bookmark, node_date_str = self.hg_log(
            ".", "{node}\n{tag}\n{bookmark}\n{date|shortdate}"
        ).split("\n")

        # TODO: support bookmarks and topics (but nowadays bookmarks are
        # mainly used to emulate Git branches, which is already supported with
        # the dedicated class GitWorkdirHgClient)

        branch, dirty_str, dirty_date = self.do(
            ["hg", "id", "-T", "{branch}\n{if(dirty, 1, 0)}\n{date|shortdate}"]
        ).split("\n")
        dirty = bool(int(dirty_str))
        # todo: fromiso
        node_date = datetime.date(
            *map(int, (dirty_date if dirty else node_date_str).split("-"))
        )

        if node.count("0") == len(node):
            trace("initial node", self.path)
            return meta(
                "0.0", config=config, dirty=dirty, branch=branch, node_date=node_date
            )

        node = "h" + node[:7]

        tags = tags_str.split()
        if "tip" in tags:
            # tip is not a real tag
            tags.remove("tip")

        if tags:
            tag = tag_to_version(tags[0])
            if tag:
                return meta(tag, dirty=dirty, branch=branch, config=config)

        try:
            tag_str = self.get_latest_normalizable_tag()
            if tag_str is None:
                dist = self.get_distance_revs("")
            else:
                dist = self.get_distance_revs(tag_str)

            if tag_str == "null" or tag_str is None:
                tag = Version("0.0")
                dist = int(dist) + 1
            else:
                tag = tag_to_version(tag_str, config=config)
                assert tag is not None

            if self.check_changes_since_tag(tag_str) or dirty:
                return meta(
                    tag,
                    distance=dist,
                    node=node,
                    dirty=dirty,
                    branch=branch,
                    config=config,
                    node_date=node_date,
                )
            else:
                return meta(tag, config=config, node_date=node_date)

        except ValueError as e:
            trace("error", e)
            pass  # unpacking failed, old hg

        return None

    def hg_log(self, revset: str, template: str) -> str:
        cmd = ["hg", "log", "-r", revset, "-T", template]
        return self.do(cmd)

    def get_latest_normalizable_tag(self) -> str | None:
        # Gets all tags containing a '.' (see #229) from oldest to newest
        outlines = self.hg_log(
            revset="ancestors(.) and tag('re:\\.')",
            template="{tags}{if(tags, '\n', '')}",
        ).split()
        if not outlines:
            return None
        tag = outlines[-1].split()[-1]
        return tag

    def get_distance_revs(self, rev1: str, rev2: str = ".") -> int:

        revset = f"({rev1}::{rev2})"
        out = self.hg_log(revset, ".")
        return len(out) - 1

    def check_changes_since_tag(self, tag: str | None) -> bool:

        if tag == "0.0" or tag is None:
            return True

        revset = (
            "(branch(.)"  # look for revisions in this branch only
            f" and tag({tag!r})::."  # after the last tag
            # ignore commits that only modify .hgtags and nothing else:
            " and (merge() or file('re:^(?!\\.hgtags).*$'))"
            f" and not tag({tag!r}))"  # ignore the tagged commit itself
        )

        return bool(self.hg_log(revset, "."))


def parse(root: _t.PathT, config: Configuration | None = None) -> ScmVersion | None:
    if not config:
        config = Configuration(root=root)

    if os.path.exists(os.path.join(root, ".hg/git")):
        paths, _, ret = do_ex("hg path", root)
        if not ret:
            for line in paths.split("\n"):
                if line.startswith("default ="):
                    path = Path(line.split()[2])
                    if path.name.endswith(".git") or (path / ".git").exists():
                        from .git import _git_parse_inner
                        from .hg_git import GitWorkdirHgClient

                        wd_hggit = GitWorkdirHgClient.from_potential_worktree(root)
                        if wd_hggit:
                            return _git_parse_inner(config, wd_hggit)

    wd = HgWorkdir.from_potential_worktree(config.absolute_root)

    if wd is None:
        return None

    return wd.get_meta(config)


def archival_to_version(
    data: dict[str, str], config: Configuration | None = None
) -> ScmVersion:
    trace("data", data)
    node = data.get("node", "")[:12]
    if node:
        node = "h" + node
    if "tag" in data:
        return meta(data["tag"], config=config)
    elif "latesttag" in data:
        return meta(
            data["latesttag"],
            distance=int(data["latesttagdistance"]),
            node=node,
            config=config,
        )
    else:
        return meta("0.0", node=node, config=config)


def parse_archival(root: _t.PathT, config: Configuration | None = None) -> ScmVersion:
    archival = os.path.join(root, ".hg_archival.txt")
    data = data_from_mime(archival)
    return archival_to_version(data, config=config)
