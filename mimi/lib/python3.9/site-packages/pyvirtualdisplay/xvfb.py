import logging

from pyvirtualdisplay.abstractdisplay import AbstractDisplay

log = logging.getLogger(__name__)

PROGRAM = "Xvfb"


class XvfbDisplay(AbstractDisplay):
    """
    Xvfb wrapper

    Xvfb is an X server that can run on machines with no display
    hardware and no physical input devices. It emulates a dumb
    framebuffer using virtual memory.
    """

    def __init__(
        self,
        size=(1024, 768),
        color_depth=24,
        bgcolor="black",
        use_xauth=False,
        fbdir=None,
        dpi=None,
        retries=10,
        extra_args=[],
        manage_global_env=True,
    ):
        """
        :param bgcolor: 'black' or 'white'
        :param fbdir: If non-null, the virtual screen is memory-mapped
            to a file in the given directory ('-fbdir' option)
        :param dpi: screen resolution in dots per inch if not None
        """
        self._screen = 0
        self._size = size
        self._color_depth = color_depth
        self._bgcolor = bgcolor
        self._fbdir = fbdir
        self._dpi = dpi

        AbstractDisplay.__init__(
            self,
            PROGRAM,
            use_xauth=use_xauth,
            retries=retries,
            extra_args=extra_args,
            manage_global_env=manage_global_env,
        )

    def _check_flags(self, helptext):
        pass

    def _cmd(self):
        cmd = [
            dict(black="-br", white="-wr")[self._bgcolor],
            "-nolisten",
            "tcp",
            "-screen",
            str(self._screen),
            "x".join(map(str, list(self._size) + [self._color_depth])),
        ]
        if self._fbdir:
            cmd += ["-fbdir", self._fbdir]
        if self._dpi is not None:
            cmd += ["-dpi", str(self._dpi)]
        if self._has_displayfd:
            cmd += ["-displayfd", str(self._pipe_wfd)]
        else:
            cmd += [self.new_display_var]
        return [PROGRAM] + cmd
