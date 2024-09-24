import fnmatch
import logging
import os
import select
import subprocess
import tempfile
import time
from threading import Lock

from pyvirtualdisplay import xauth
from pyvirtualdisplay.util import get_helptext, platform_is_osx

log = logging.getLogger(__name__)

# try:
#     import fcntl
# except ImportError:
#     fcntl = None

_mutex = Lock()
_mutex_popen = Lock()


_MIN_DISPLAY_NR = 1000
_USED_DISPLAY_NR_LIST = []

_X_START_TIMEOUT = 10
_X_START_TIME_STEP = 0.1
_X_START_WAIT = 0.1


class XStartTimeoutError(Exception):
    pass


class XStartError(Exception):
    pass


def _lock_files():
    tmpdir = "/tmp"
    try:
        ls = os.listdir(tmpdir)
    except FileNotFoundError:
        log.warning("missing /tmp")
        return []
    pattern = ".X*-lock"
    names = fnmatch.filter(ls, pattern)
    ls = [os.path.join(tmpdir, child) for child in names]
    ls = [p for p in ls if os.path.isfile(p)]
    return ls


def _search_for_display():
    # search for free display
    ls = list(map(lambda x: int(x.split("X")[1].split("-")[0]), _lock_files()))
    if len(ls):
        display = max(_MIN_DISPLAY_NR, max(ls) + 3)
    else:
        display = _MIN_DISPLAY_NR

    return display


class AbstractDisplay(object):
    """
    Common parent for X servers (Xvfb,Xephyr,Xvnc)
    """

    def __init__(self, program, use_xauth, retries, extra_args, manage_global_env):
        self._extra_args = extra_args
        self._retries = retries
        self._program = program
        self.stdout = None
        self.stderr = None
        self.old_display_var = None
        self._subproc = None
        self.display = None
        self._is_started = False
        self._manage_global_env = manage_global_env
        self._reset_global_env = False
        self._pipe_wfd = None
        self._retries_current = 0

        helptext = get_helptext(program)
        self._has_displayfd = "-displayfd" in helptext
        if not self._has_displayfd:
            log.debug("-displayfd flag is missing.")
        PYVIRTUALDISPLAY_DISPLAYFD = os.environ.get("PYVIRTUALDISPLAY_DISPLAYFD")
        if PYVIRTUALDISPLAY_DISPLAYFD:
            log.debug("PYVIRTUALDISPLAY_DISPLAYFD=%s", PYVIRTUALDISPLAY_DISPLAYFD)
            # '0'->false, '1'->true
            self._has_displayfd = bool(int(PYVIRTUALDISPLAY_DISPLAYFD))
        else:
            # TODO: macos: displayfd is available on XQuartz-2.7.11 but it doesn't work, always 0 is returned
            if platform_is_osx():
                self._has_displayfd = False

        self._check_flags(helptext)

        if use_xauth and not xauth.is_installed():
            raise xauth.NotFoundError()

        self._use_xauth = use_xauth
        self._old_xauth = None
        self._xauth_filename = None

    def _check_flags(self, helptext):
        pass

    def _cmd(self):
        raise NotImplementedError()

    def _redirect_display(self, on):
        """
        on:
         * True -> set $DISPLAY to virtual screen
         * False -> set $DISPLAY to original screen

        :param on: bool
        """
        d = self.new_display_var if on else self.old_display_var
        if d is None:
            log.debug("unset $DISPLAY")
            try:
                del os.environ["DISPLAY"]
            except KeyError:
                log.warning("$DISPLAY was already unset.")
        else:
            log.debug("set $DISPLAY=%s", d)
            os.environ["DISPLAY"] = d

    def _env(self):
        env = os.environ.copy()
        env["DISPLAY"] = self.new_display_var
        return env

    def start(self):
        """
        start display

        :rtype: self
        """
        if self._is_started:
            raise XStartError(self, "Display was started twice.")
        self._is_started = True

        if self._has_displayfd:
            self._start1_has_displayfd()
        else:
            i = 0
            while True:
                self._retries_current = i + 1
                try:
                    self._start1()
                    break
                except XStartError:
                    log.warning("start failed %s", i + 1)
                    time.sleep(0.05)
                    i += 1
                    if i >= self._retries:
                        raise XStartError(
                            "No success after %s retries. Last stderr: %s"
                            % (self._retries, self.stderr)
                        )
        if self._manage_global_env:
            self._redirect_display(True)
            self._reset_global_env = True

    def _popen(self, use_pass_fds):
        with _mutex_popen:
            if use_pass_fds:
                self._subproc = subprocess.Popen(
                    self._command,
                    pass_fds=[self._pipe_wfd],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=False,
                )
            else:
                self._subproc = subprocess.Popen(
                    self._command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=False,
                )

    def _start1_has_displayfd(self):
        # stdout doesn't work on osx -> create own pipe
        rfd, self._pipe_wfd = os.pipe()

        self._command = self._cmd() + self._extra_args
        log.debug("command: %s", self._command)

        self._popen(use_pass_fds=True)

        self.display = int(self._wait_for_pipe_text(rfd))
        os.close(rfd)
        os.close(self._pipe_wfd)

        self.new_display_var = ":%s" % int(self.display)

        if self._use_xauth:
            self._setup_xauth()

        # https://github.com/ponty/PyVirtualDisplay/issues/2
        # https://github.com/ponty/PyVirtualDisplay/issues/14
        self.old_display_var = os.environ.get("DISPLAY", None)

    def _start1(self):
        with _mutex:
            self.display = _search_for_display()
            while self.display in _USED_DISPLAY_NR_LIST:
                self.display += 1
            self.new_display_var = ":%s" % int(self.display)

            _USED_DISPLAY_NR_LIST.append(self.display)

        self._command = self._cmd() + self._extra_args
        log.debug("command: %s", self._command)

        self._popen(use_pass_fds=False)

        self.new_display_var = ":%s" % int(self.display)

        if self._use_xauth:
            self._setup_xauth()

        # https://github.com/ponty/PyVirtualDisplay/issues/2
        # https://github.com/ponty/PyVirtualDisplay/issues/14
        self.old_display_var = os.environ.get("DISPLAY", None)

        # wait until X server is active
        start_time = time.time()

        d = self.new_display_var
        ok = False
        time.sleep(0.05)  # give time for early exit
        while True:
            if not self.is_alive():
                break

            try:
                xdpyinfo = subprocess.Popen(
                    ["xdpyinfo"],
                    env=self._env(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=False,
                )
                _, _ = xdpyinfo.communicate()
                exit_code = xdpyinfo.returncode
            except FileNotFoundError:
                log.warning(
                    "xdpyinfo was not found, X start can not be checked! Please install xdpyinfo!"
                )
                time.sleep(_X_START_WAIT)  # old method
                ok = True
                break
            # try:
            #     xdpyinfo = EasyProcess(["xdpyinfo"], env=self._env())
            #     xdpyinfo.enable_stdout_log = False
            #     xdpyinfo.enable_stderr_log = False
            #     exit_code = xdpyinfo.call().return_code
            # except EasyProcessError:
            #     log.warning(
            #         "xdpyinfo was not found, X start can not be checked! Please install xdpyinfo!"
            #     )
            #     time.sleep(_X_START_WAIT)  # old method
            #     ok = True
            #     break

            if exit_code != 0:
                pass
            else:
                log.info('Successfully started X with display "%s".', d)
                ok = True
                break

            if time.time() - start_time >= _X_START_TIMEOUT:
                break
            time.sleep(_X_START_TIME_STEP)
        if not self.is_alive():
            log.warning("process exited early. stderr:%s", self.stderr)
            msg = "Failed to start process: %s"
            raise XStartError(msg % self)
        if not ok:
            msg = 'Failed to start X on display "%s" (xdpyinfo check failed, stderr:[%s]).'
            raise XStartTimeoutError(msg % (d, xdpyinfo.stderr))

    def _wait_for_pipe_text(self, rfd):
        s = ""
        start_time = time.time()
        while True:
            (rfd_changed_ls, _, _) = select.select([rfd], [], [], 0.1)
            if not self.is_alive():
                raise XStartError(
                    "%s program closed. command: %s stderr: %s"
                    % (self._program, self._command, self.stderr)
                )
            if rfd in rfd_changed_ls:
                c = os.read(rfd, 1)
                if c == b"\n":
                    break
                s += c.decode("ascii")

            # this timeout is for "eternal" hang. see #62
            if time.time() - start_time >= 600:  # = 10 minutes
                raise XStartTimeoutError(
                    "No reply from program %s. command:%s"
                    % (
                        self._program,
                        self._command,
                    )
                )
        return s

    def stop(self):
        """
        stop display

        :rtype: self
        """
        if not self._is_started:
            raise XStartError("stop() is called before start().")

        if self._reset_global_env:
            self._redirect_display(False)

        if self.is_alive():
            try:
                self._subproc.kill()
            except OSError as oserror:
                log.debug("exception in terminate:%s", oserror)

            self._subproc.wait()
            self._read_stdout_stderr()
        if self._use_xauth:
            self._clear_xauth()
        return self

    def _read_stdout_stderr(self):
        if self.stdout is None:
            (self.stdout, self.stderr) = self._subproc.communicate()

            log.debug("stdout=%s", self.stdout)
            log.debug("stderr=%s", self.stderr)

    def _setup_xauth(self):
        """
        Set up the Xauthority file and the XAUTHORITY environment variable.
        """
        handle, filename = tempfile.mkstemp(
            prefix="PyVirtualDisplay.", suffix=".Xauthority"
        )
        self._xauth_filename = filename
        os.close(handle)
        # Save old environment
        self._old_xauth = {}
        self._old_xauth["AUTHFILE"] = os.getenv("AUTHFILE")
        self._old_xauth["XAUTHORITY"] = os.getenv("XAUTHORITY")

        os.environ["AUTHFILE"] = os.environ["XAUTHORITY"] = filename
        cookie = xauth.generate_mcookie()
        xauth.call("add", self.new_display_var, ".", cookie)

    def _clear_xauth(self):
        """
        Clear the Xauthority file and restore the environment variables.
        """
        os.remove(self._xauth_filename)
        for varname in ["AUTHFILE", "XAUTHORITY"]:
            if self._old_xauth[varname] is None:
                del os.environ[varname]
            else:
                os.environ[varname] = self._old_xauth[varname]
        self._old_xauth = None

    def __enter__(self):
        """used by the :keyword:`with` statement"""
        self.start()
        return self

    def __exit__(self, *exc_info):
        """used by the :keyword:`with` statement"""
        self.stop()

    def is_alive(self):
        if not self._subproc:
            return False
        # return self.return_code is None
        rc = self._subproc.poll()
        if rc is not None:
            # proc exited
            self._read_stdout_stderr()
        return rc is None

    # @property
    # def return_code(self):
    #     if not self._subproc:
    #         return None
    #     rc = self._subproc.poll()
    #     if rc is not None:
    #         # proc exited
    #         self._read_stdout_stderr()
    #     return rc

    @property
    def pid(self):
        """
        PID (:attr:`subprocess.Popen.pid`)

        :rtype: int
        """
        if self._subproc:
            return self._subproc.pid
