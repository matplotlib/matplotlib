from __future__ import annotations

import os
from contextlib import suppress
from datetime import date
from datetime import datetime

from . import _types as _t
from .git import GitWorkdir
from .hg import HgWorkdir
from .utils import _CmdResult
from .utils import do_ex
from .utils import require_command
from .utils import trace


_FAKE_GIT_DESCRIBE_ERROR = _CmdResult("<>hg git failed", "", 1)


class GitWorkdirHgClient(GitWorkdir, HgWorkdir):
    COMMAND = "hg"

    @classmethod
    def from_potential_worktree(cls, wd: _t.PathT) -> GitWorkdirHgClient | None:
        require_command(cls.COMMAND)
        root, _, ret = do_ex(["hg", "root"], wd)
        if ret:
            return None
        return cls(root)

    def is_dirty(self) -> bool:
        out, _, _ = self.do_ex('hg id -T "{dirty}"')
        return bool(out)

    def get_branch(self) -> str | None:
        res = self.do_ex('hg id -T "{bookmarks}"')
        if res.returncode:
            trace("branch err", res)
            return None
        return res.out

    def get_head_date(self) -> date | None:
        date_part, err, ret = self.do_ex('hg log -r . -T "{shortdate(date)}"')
        if ret:
            trace("head date err", date_part, err, ret)
            return None
        return datetime.strptime(date_part, r"%Y-%m-%d").date()

    def is_shallow(self) -> bool:
        return False

    def fetch_shallow(self) -> None:
        pass

    def get_hg_node(self) -> str | None:
        node, _, ret = self.do_ex('hg log -r . -T "{node}"')
        if not ret:
            return node
        else:
            return None

    def _hg2git(self, hg_node: str) -> str | None:
        with suppress(FileNotFoundError):
            with open(os.path.join(self.path, ".hg/git-mapfile")) as map_items:
                for item in map_items:
                    if hg_node in item:
                        git_node, hg_node = item.split()
                        return git_node
        return None

    def node(self) -> str | None:
        hg_node = self.get_hg_node()
        if hg_node is None:
            return None

        git_node = self._hg2git(hg_node)

        if git_node is None:
            # trying again after hg -> git
            self.do_ex("hg gexport")
            git_node = self._hg2git(hg_node)

            if git_node is None:
                trace("Cannot get git node so we use hg node", hg_node)

                if hg_node == "0" * len(hg_node):
                    # mimic Git behavior
                    return None

                return hg_node

        return git_node[:7]

    def count_all_nodes(self) -> int:
        revs, _, _ = self.do_ex(["hg", "log", "-r", "ancestors(.)", "-T", "."])
        return len(revs)

    def default_describe(self) -> _CmdResult:
        """
        Tentative to reproduce the output of

        `git describe --dirty --tags --long --match *[0-9]*`

        """
        hg_tags_str, _, ret = self.do_ex(
            [
                "hg",
                "log",
                "-r",
                "(reverse(ancestors(.)) and tag(r're:v?[0-9].*'))",
                "-T",
                "{tags}{if(tags, ' ', '')}",
            ]
        )
        if ret:
            return _FAKE_GIT_DESCRIBE_ERROR
        hg_tags: list[str] = hg_tags_str.split()

        if not hg_tags:
            return _FAKE_GIT_DESCRIBE_ERROR

        with open(os.path.join(self.path, ".hg/git-tags")) as fp:
            git_tags: dict[str, str] = dict(line.split()[::-1] for line in fp)

        tag: str
        for hg_tag in hg_tags:
            if hg_tag in git_tags:
                tag = hg_tag
                break
        else:
            trace("tag not found", hg_tags, git_tags)
            return _FAKE_GIT_DESCRIBE_ERROR

        out, _, ret = self.do_ex(["hg", "log", "-r", f"'{tag}'::.", "-T", "."])
        if ret:
            return _FAKE_GIT_DESCRIBE_ERROR
        distance = len(out) - 1

        node = self.node()
        assert node is not None
        desc = f"{tag}-{distance}-g{node}"

        if self.is_dirty():
            desc += "-dirty"
        trace("desc", desc)
        return _CmdResult(desc, "", 0)
