#!/usr/bin/env python
"""
Use matplotlib interactively from the prompt

This script is from
http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/65109 by Brian
McErlean and John Finlay, with minor modifications for matplotlib and
win32 usage.

"""
import __builtin__
import __main__
import codeop
import keyword
import os
import re
try:
    import readline
except ImportError:
    haveReadline = 0
else:
    haveReadline = 1

import threading
import traceback
import signal
import sys

import pygtk
pygtk.require("2.0")
import gtk

from matplotlib.cbook import wrap
from matplotlib.matlab import *
import matplotlib.matlab

def walk_class (klass):
    list = []
    for item in dir (klass.__class__):
        if item[0] != "_":
            list.append (item)

    for base in klass.__class__.__bases__:
        list = list + walk_class (base())

    return list

class Completer:
    def __init__ (self, lokals):
        self.locals = lokals

        self.completions = keyword.kwlist + \
                           __builtin__.__dict__.keys() + \
                           __main__.__dict__.keys()
    def complete (self, text, state):
        if state == 0:
            if "." in text:
                self.matches = self.attr_matches (text)
            else:
                self.matches = self.global_matches (text)
        try:
            return self.matches[state]
        except IndexError:
            return None

    def update (self, locs):
        self.locals = locs

        for key in self.locals.keys ():
            if not key in self.completions:
                self.completions.append (key)

    def global_matches (self, text):
        matches = []
        n = len (text)
        for word in self.completions:
            if word[:n] == text:
                matches.append (word)
        return matches

    def attr_matches (self, text):
        m = re.match(r"(\w+(\.\w+)*)\.(\w*)", text)
        if not m:
            return
        expr, attr = m.group(1, 3)

        obj = eval (expr, self.locals)
        if str (obj)[1:4] == "gtk":
            words = walk_class (obj)
        else:
            words = dir(eval(expr, self.locals))
            
        matches = []
        n = len(attr)
        for word in words:
            if word[:n] == attr:
                matches.append ("%s.%s" % (expr, word))
        return matches

class GtkInterpreter (threading.Thread):
    """Run a gtk mainloop() in a separate thread.
    Python commands can be passed to the thread where they will be executed.
    This is implemented by periodically checking for passed code using a
    GTK timeout callback.
    """
    TIMEOUT = 100 # Millisecond interval between timeouts.
    
    def __init__ (self):
        threading.Thread.__init__ (self)
        self.ready = threading.Condition ()
        self.globs = globals ()
        self.locs = locals ()
        self._kill = 0
        self.cmd = ''       # Current code block
        self.new_cmd = None # Waiting line of code, or None if none waiting

        self.completer = Completer (self.locs)
        if haveReadline:
            readline.set_completer (self.completer.complete)
            readline.parse_and_bind ('tab: complete')

    def run (self):
        gtk.timeout_add (self.TIMEOUT, self.code_exec)
        try:
            if gtk.gtk_version[0] == 2:
                gtk.threads_init()
        except:
            pass

        gtk.main ()

    def code_exec (self):
        """Execute waiting code.  Called every timeout period."""
        self.ready.acquire ()
        if self._kill: gtk.main_quit ()
        if self.new_cmd != None:  
            self.ready.notify ()  
            self.cmd = self.cmd + self.new_cmd
            self.new_cmd = None
            try:
                tmp = self.cmd[:-1]
                code = codeop.compile_command (self.cmd[:-1]) 
                if code: 
                    self.cmd = ''
                    #print 'Execing', tmp
                    exec (code, self.globs, self.locs)
                    self.completer.update (self.locs)
            except Exception:
                traceback.print_exc ()
                self.cmd = ''  
                                    
        self.ready.release()
        return 1 
            
    def feed (self, code):
        """Feed a line of code to the thread.
        This function will block until the code checked by the GTK thread.
        Return true if executed the code.
        Returns false if deferring execution until complete block available.
        """
        if (not code) or (code[-1]<>'\n'): code = code +'\n' # raw_input strips newline
        self.completer.update (self.locs) 
        self.ready.acquire()
        self.new_cmd = code
        self.ready.wait ()  # Wait until processed in timeout interval
        self.ready.release ()
        
        return not self.cmd

    def kill (self):
        """Kill the thread, returning when it has been shut down."""
        self.ready.acquire()
        self._kill=1
        self.ready.release()
        self.join()
        
# Read user input in a loop, and send each line to the interpreter thread.

def signal_handler (*args):
    print "SIGNAL:", args
    sys.exit()

if __name__=="__main__":
    signal.signal (signal.SIGINT, signal_handler)
    signal.signal (signal.SIGSEGV, signal_handler)
    
    prompt = '>> '
    interpreter = GtkInterpreter ()
    interpreter.start ()
    interpreter.feed("import matplotlib")
    interpreter.feed("matplotlib.interactive(1)")
    interpreter.feed("from matplotlib.matlab import *")

    # turn off rendering until end of script
    matplotlib.matlab.interactive = 0
    print sys.argv
    if len (sys.argv) > 1:
        try: inFile = file(sys.argv[1], 'r')
        except IOError: pass
        else:
            for line in file(sys.argv[1], 'r'):
                if line.lstrip().find('show()')==0: continue
                print '>>', line.rstrip()
                interpreter.feed(line)
        #gcf().draw()
    print """Welcome to matplotlib.

    help(matlab)   -- shows a list of all matlab compatible commands provided
    help(plotting) -- shows a list of plot specific commands
    """ 


    try:
        while 1:
	    command = raw_input (prompt) + '\n' # raw_input strips newlines
            prompt = interpreter.feed (command) and '>> ' or '... '
    except (EOFError, KeyboardInterrupt): pass

    interpreter.kill()
    print



