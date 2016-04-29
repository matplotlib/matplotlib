"""Multithreaded interactive interpreter with GTK and Matplotlib support.

WARNING:
As of 2010/06/25, this is not working, at least on Linux.
I have disabled it as a runnable script. - EF


Usage:

  pyint-gtk.py -> starts shell with gtk thread running separately

  pyint-gtk.py -pylab [filename] -> initializes matplotlib, optionally running
  the named file.  The shell starts after the file is executed.

Threading code taken from:
http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/65109, by Brian
McErlean and John Finlay.

Matplotlib support taken from interactive.py in the matplotlib distribution.

Also borrows liberally from code.py in the Python standard library."""

from __future__ import print_function

__author__ = "Fernando Perez <Fernando.Perez@colorado.edu>"

import sys
import code
import threading

import gobject
import gtk

try:
    import readline
except ImportError:
    has_readline = False
else:
    has_readline = True


class MTConsole(code.InteractiveConsole):
    """Simple multi-threaded shell"""

    def __init__(self, on_kill=None, *args, **kw):
        code.InteractiveConsole.__init__(self, *args, **kw)
        self.code_to_run = None
        self.ready = threading.Condition()
        self._kill = False
        if on_kill is None:
            on_kill = []
        # Check that all things to kill are callable:
        for _ in on_kill:
            if not callable(_):
                raise TypeError('on_kill must be a list of callables')
        self.on_kill = on_kill
        # Set up tab-completer
        if has_readline:
            import rlcompleter
            try:  # this form only works with python 2.3
                self.completer = rlcompleter.Completer(self.locals)
            except:  # simpler for py2.2
                self.completer = rlcompleter.Completer()

            readline.set_completer(self.completer.complete)
            # Use tab for completions
            readline.parse_and_bind('tab: complete')
            # This forces readline to automatically print the above list when tab
            # completion is set to 'complete'.
            readline.parse_and_bind('set show-all-if-ambiguous on')
            # Bindings for incremental searches in the history. These searches
            # use the string typed so far on the command line and search
            # anything in the previous input history containing them.
            readline.parse_and_bind('"\C-r": reverse-search-history')
            readline.parse_and_bind('"\C-s": forward-search-history')

    def runsource(self, source, filename="<input>", symbol="single"):
        """Compile and run some source in the interpreter.

        Arguments are as for compile_command().

        One several things can happen:

        1) The input is incorrect; compile_command() raised an
        exception (SyntaxError or OverflowError).  A syntax traceback
        will be printed by calling the showsyntaxerror() method.

        2) The input is incomplete, and more input is required;
        compile_command() returned None.  Nothing happens.

        3) The input is complete; compile_command() returned a code
        object.  The code is executed by calling self.runcode() (which
        also handles run-time exceptions, except for SystemExit).

        The return value is True in case 2, False in the other cases (unless
        an exception is raised).  The return value can be used to
        decide whether to use sys.ps1 or sys.ps2 to prompt the next
        line.
        """
        try:
            code = self.compile(source, filename, symbol)
        except (OverflowError, SyntaxError, ValueError):
            # Case 1
            self.showsyntaxerror(filename)
            return False

        if code is None:
            # Case 2
            return True

        # Case 3
        # Store code in self, so the execution thread can handle it
        self.ready.acquire()
        self.code_to_run = code
        self.ready.wait()  # Wait until processed in timeout interval
        self.ready.release()

        return False

    def runcode(self):
        """Execute a code object.

        When an exception occurs, self.showtraceback() is called to display a
        traceback."""

        self.ready.acquire()
        if self._kill:
            print('Closing threads...')
            sys.stdout.flush()
            for tokill in self.on_kill:
                tokill()
            print('Done.')

        if self.code_to_run is not None:
            self.ready.notify()
            code.InteractiveConsole.runcode(self, self.code_to_run)

        self.code_to_run = None
        self.ready.release()
        return True

    def kill(self):
        """Kill the thread, returning when it has been shut down."""
        self.ready.acquire()
        self._kill = True
        self.ready.release()


class GTKInterpreter(threading.Thread):
    """Run gtk.main in the main thread and a python interpreter in a
    separate thread.
    Python commands can be passed to the thread where they will be executed.
    This is implemented by periodically checking for passed code using a
    GTK timeout callback.
    """
    TIMEOUT = 100  # Millisecond interval between timeouts.

    def __init__(self, banner=None):
        threading.Thread.__init__(self)
        self.banner = banner
        self.shell = MTConsole(on_kill=[gtk.main_quit])

    def run(self):
        self.pre_interact()
        self.shell.interact(self.banner)
        self.shell.kill()

    def mainloop(self):
        self.start()
        gobject.timeout_add(self.TIMEOUT, self.shell.runcode)
        try:
            if gtk.gtk_version[0] >= 2:
                gtk.gdk.threads_init()
        except AttributeError:
            pass
        gtk.main()
        self.join()

    def pre_interact(self):
        """This method should be overridden by subclasses.

        It gets called right before interact(), but after the thread starts.
        Typically used to push initialization code into the interpreter"""

        pass


class MatplotLibInterpreter(GTKInterpreter):
    """Threaded interpreter with matplotlib support.

    Note that this explicitly sets GTKAgg as the backend, since it has
    specific GTK hooks in it."""

    def __init__(self, banner=None):
        banner = """\nWelcome to matplotlib, a MATLAB-like python environment.
    help(matlab)   -> help on matlab compatible commands from matplotlib.
    help(plotting) -> help on plotting commands.
    """
        GTKInterpreter.__init__(self, banner)

    def pre_interact(self):
        """Initialize matplotlib before user interaction begins"""

        push = self.shell.push
        # Code to execute in user's namespace
        lines = ["import matplotlib",
                 "matplotlib.use('GTKAgg')",
                 "matplotlib.interactive(1)",
                 "import matplotlib.pylab as pylab",
                 "from matplotlib.pylab import *\n"]

        map(push, lines)

        # Execute file if given.
        if len(sys.argv) > 1:
            import matplotlib
            matplotlib.interactive(0)  # turn off interaction
            fname = sys.argv[1]
            try:
                inFile = file(fname, 'r')
            except IOError:
                print('*** ERROR *** Could not read file <%s>' % fname)
            else:
                print('*** Executing file <%s>:' % fname)
                for line in inFile:
                    if line.lstrip().find('show()') == 0:
                        continue
                    print('>>', line)
                    push(line)
                inFile.close()
            matplotlib.interactive(1)   # turn on interaction

if __name__ == '__main__':
    print("This demo is not presently functional, so running")
    print("it as a script has been disabled.")
    sys.exit()
    # Quick sys.argv hack to extract the option and leave filenames in sys.argv.
    # For real option handling, use optparse or getopt.
    if len(sys.argv) > 1 and sys.argv[1] == '-pylab':
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        MatplotLibInterpreter().mainloop()
    else:
        GTKInterpreter().mainloop()
