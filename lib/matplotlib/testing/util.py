import subprocess
import sys


class MiniExpect:
    """
    This is a very basic version of pexpect, providing only the
    functionality necessary for the testing framework, built on top of
    `subprocess` rather than directly on lower-level calls.
    """
    def __init__(self, args):
        """
        Start the subprocess so it may start accepting commands.

        *args* is a list of commandline arguments to pass to
        `subprocess.Popen`.
        """
        self._name = args[0]
        self._process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)

    def check_alive(self):
        """
        Raises a RuntimeError if the process is no longer alive.
        """
        returncode = self._process.poll()
        if returncode is not None:
            raise RuntimeError("%s unexpectedly quit" % self._name)

    def sendline(self, line):
        """
        Send a line to the process.
        """
        line = line.encode('ascii')
        self.check_alive()
        stdin = self._process.stdin
        stdin.write(line)
        stdin.write(b'\n')
        stdin.flush()

    def expect(self, s, output=None):
        """
        Wait for the string *s* to appear in the child process's output.

        *output* (optional) is a writable file object where all of the
        content preceding *s* will be written.
        """
        s = s.encode('ascii')
        read = self._process.stdout.read
        pos = 0
        buf = b''
        while True:
            self.check_alive()
            char = read(1)
            if not char:
                raise IOError("Unexpected end-of-file")

            if sys.version_info[0] >= 3:
                match = (ord(char) == s[pos])
            else:
                match = (char == s[pos])

            if match:
                buf += char
                pos += 1
                if pos == len(s):
                    return
            else:
                if output is not None:
                    output.write(buf)
                    output.write(char)
                buf = b''
                pos = 0
