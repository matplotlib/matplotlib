import logging

from pyvirtualdisplay.abstractdisplay import AbstractDisplay

log = logging.getLogger(__name__)

PROGRAM = "Xephyr"


class XephyrDisplay(AbstractDisplay):
    """
    Xephyr wrapper

    Xephyr is an X server outputting to a window on a pre-existing X display
    """

    def __init__(
        self,
        size=(1024, 768),
        color_depth=24,
        bgcolor="black",
        use_xauth=False,
        retries=10,
        extra_args=[],
        manage_global_env=True,
        parent=None,
    ):
        """
        :param bgcolor: 'black' or 'white'
        """
        self._color_depth = color_depth
        self._size = size
        self._bgcolor = bgcolor
        self._parent = parent

        AbstractDisplay.__init__(
            self,
            PROGRAM,
            use_xauth=use_xauth,
            retries=retries,
            extra_args=extra_args,
            manage_global_env=manage_global_env,
        )

    def _check_flags(self, helptext):
        self._has_resizeable = "-resizeable" in helptext

    def _cmd(self):
        cmd = (
            [
                PROGRAM,
            ]
            + (["-parent", self._parent] if self._parent else [])
            + [
                dict(black="-br", white="-wr")[self._bgcolor],
                "-screen",
                "x".join(map(str, list(self._size) + [self._color_depth])),
            ]
        )
        if self._has_displayfd:
            cmd += ["-displayfd", str(self._pipe_wfd)]
        else:
            cmd += [self.new_display_var]

        if self._has_resizeable:
            cmd += ["-resizeable"]
        return cmd
