import sys
from unittest.mock import MagicMock


class MyCairoCffi(MagicMock):
    __name__ = "cairocffi"


class MyWX(MagicMock):
    class Panel(object):
        pass

    class ToolBar(object):
        pass

    class Frame(object):
        pass

    class StatusBar(object):
        pass


def setup(app):
    sys.modules.update(
        cairocffi=MyCairoCffi(),
        wx=MyWX(),
    )
    return {'parallel_read_safe': True, 'parallel_write_safe': True}
