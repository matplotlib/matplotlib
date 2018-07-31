import sys

try:
    from unittest.mock import MagicMock
except ImportError:
    from mock import MagicMock


class MyCairoCffi(MagicMock):
    version_info = (1, 4, 0)


class MyWX(MagicMock):
    class Panel(object):
        pass

    class ToolBar(object):
        pass

    class Frame(object):
        pass

    VERSION_STRING = '2.9'


def setup(app):
    sys.modules['cairocffi'] = MyCairoCffi()
    sys.modules['wx'] = MyWX()
    sys.modules['wxversion'] = MagicMock()

    metadata = {'parallel_read_safe': True, 'parallel_write_safe': True}
    return metadata
