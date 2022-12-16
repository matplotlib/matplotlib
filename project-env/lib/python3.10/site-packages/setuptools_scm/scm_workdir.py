from __future__ import annotations

from typing import ClassVar
from typing import TYPE_CHECKING

from .utils import _CmdResult
from .utils import do
from .utils import do_ex
from .utils import require_command

if TYPE_CHECKING:
    from . import _types as _t


class Workdir:
    COMMAND: ClassVar[str]

    def __init__(self, path: _t.PathT):
        require_command(self.COMMAND)
        self.path = path

    def do_ex(self, cmd: _t.CMD_TYPE) -> _CmdResult:
        return do_ex(cmd, cwd=self.path)

    def do(self, cmd: _t.CMD_TYPE) -> str:
        return do(cmd, cwd=self.path)
