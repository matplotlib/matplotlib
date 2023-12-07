import logging

from pyvirtualdisplay.abstractdisplay import AbstractDisplay

log = logging.getLogger(__name__)

PROGRAM = "Xvnc"


class XvncDisplay(AbstractDisplay):
    """
    Xvnc wrapper
    """

    def __init__(
        self,
        size=(1024, 768),
        color_depth=24,
        bgcolor="black",
        use_xauth=False,
        rfbport=5900,
        rfbauth=None,
        retries=10,
        extra_args=[],
        manage_global_env=True,
    ):
        """
        :param bgcolor: 'black' or 'white'
        :param rfbport: Specifies the TCP port on which Xvnc listens for connections from viewers
        (the protocol used in VNC is called RFB - "remote framebuffer").
        The default is 5900 plus the display number.
        :param rfbauth: Specifies the file containing the password used to authenticate viewers.
        """
        self._size = size
        self._color_depth = color_depth
        self._bgcolor = bgcolor
        self._rfbport = rfbport
        self._rfbauth = rfbauth

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
            PROGRAM,
            "-depth",
            str(self._color_depth),
            "-geometry",
            "%dx%d" % (self._size[0], self._size[1]),
            "-rfbport",
            str(self._rfbport),
        ]

        if self._rfbauth:
            cmd += ["-rfbauth", str(self._rfbauth)]
            # default:
            # -SecurityTypes = VncAuth
        else:
            cmd += ["-SecurityTypes", "None"]

        if self._has_displayfd:
            cmd += ["-displayfd", str(self._pipe_wfd)]
        else:
            cmd += [self.new_display_var]
        return cmd
