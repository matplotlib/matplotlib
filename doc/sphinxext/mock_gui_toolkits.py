import sys
from unittest.mock import MagicMock


class MyCairoCffi(MagicMock):
    __name__ = "cairocffi"


def setup(app):
    sys.modules.update(
        cairocffi=MyCairoCffi(),
    )
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
