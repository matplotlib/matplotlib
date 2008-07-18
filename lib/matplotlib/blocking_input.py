"""
This provides several classes used for blocking interaction with figure windows:

:class:`BlockingInput`
    creates a callable object to retrieve events in a blocking way for interactive sessions

:class:`BlockingKeyMouseInput`
    creates a callable object to retrieve key or mouse clicks in a blocking way for interactive sessions.  
    Note: Subclass of BlockingInput. Used by waitforbuttonpress

:class:`BlockingMouseInput`
    creates a callable object to retrieve mouse clicks in a blocking way for interactive sessions.  
    Note: Subclass of BlockingInput.  Used by ginput

:class:`BlockingContourLabeler`
    creates a callable object to retrieve mouse clicks in a blocking way that will then be used to place labels on a ContourSet
    Note: Subclass of BlockingMouseInput.  Used by clabel
"""

import time
import numpy as np
from matplotlib import path, verbose
from cbook import is_sequence_of_strings

class BlockingInput(object):
    """
    Class that creates a callable object to retrieve events in a
    blocking way.
    """
    def __init__(self, fig, eventslist=()):
        self.fig = fig
        assert is_sequence_of_strings(eventslist), "Requires a sequence of event name strings"
        self.eventslist = eventslist

    def on_event(self, event):
        """
        Event handler that will be passed to the current figure to
        retrieve events.
        """
        # Add a new event to list - using a separate function is
        # overkill for the base class, but this is consistent with
        # subclasses
        self.add_event(event)

        verbose.report("Event %i" % len(self.events))

        # This will extract info from events
        self.post_event()
        
        # Check if we have enough events already
        if len(self.events) >= self.n and self.n > 0:
            self.done = True

    def post_event(self):
        """For baseclass, do nothing but collect events"""
        pass

    def cleanup(self):
        """Disconnect all callbacks"""
        for cb in self.callbacks:
            self.fig.canvas.mpl_disconnect(cb)

        self.callbacks=[]

    def add_event(self,event):
        """For base class, this just appends an event to events."""
        self.events.append(event)

    def pop_event(self,index=-1):
        """
        This removes an event from the event list.  Defaults to
        removing last event, but an index can be supplied.  Note that
        this does not check that there are events, much like the
        normal pop method.  If not events exist, this will throw an
        exception.
        """
        self.events.pop(index)

    def pop(self,index=-1):
        self.pop_event(index)
    pop.__doc__=pop_event.__doc__

    def __call__(self, n=1, timeout=30 ):
        """
        Blocking call to retrieve n events
        """
        
        assert isinstance(n, int), "Requires an integer argument"
        self.n = n

        self.events = []
        self.done = False
        self.callbacks = []

        # Ensure that the figure is shown
        self.fig.show()
        
        # connect the events to the on_event function call
        for n in self.eventslist:
            self.callbacks.append( self.fig.canvas.mpl_connect(n, self.on_event) )

        try:
            # wait for n clicks
            counter = 0
            while not self.done:
                self.fig.canvas.flush_events()
                time.sleep(0.01)

                # check for a timeout
                counter += 1
                if timeout > 0 and counter > timeout/0.01:
                    print "Timeout reached";
                    break;
        finally: # Run even on exception like ctrl-c
            # Disconnect the callbacks
            self.cleanup()

        # Return the events in this case
        return self.events

class BlockingMouseInput(BlockingInput):
    """
    Class that creates a callable object to retrieve mouse clicks in a
    blocking way.
    """
    def __init__(self, fig):
        BlockingInput.__init__(self, fig=fig, 
                               eventslist=('button_press_event',) )

    def post_event(self):
        """
        This will be called to process events
        """
        assert len(self.events)>0, "No events yet"

        event = self.events[-1]
        button = event.button

        # Using additional methods for each button is a bit overkill
        # for this class, but it allows for easy overloading.  Also,
        # this would make it easy to attach other type of non-mouse
        # events to these "mouse" actions.  For example, the matlab
        # version of ginput also allows you to add points with
        # keyboard clicks.  This could easily be added to this class
        # with relatively minor modification to post_event and
        # __init__.
        if button == 3:
            self.button3(event)
        elif button == 2:
            self.button2(event)
        else:
            self.button1(event)

    def button1( self, event ):
        """
        Will be called for any event involving a button other than
        button 2 or 3.  This will add a click if it is inside axes.
        """
        if event.inaxes:
            self.add_click(event)
        else: # If not a valid click, remove from event list
            BlockingInput.pop(self)

    def button2( self, event ):
        """
        Will be called for any event involving button 2.
        Button 2 ends blocking input.
        """

        # Remove last event just for cleanliness
        BlockingInput.pop(self)

        # This will exit even if not in infinite mode.  This is
        # consistent with matlab and sometimes quite useful, but will
        # require the user to test how many points were actually
        # returned before using data.
        self.done = True

    def button3( self, event ):
        """
        Will be called for any event involving button 3.
        Button 3 removes the last click.
        """
        # Remove this last event
        BlockingInput.pop(self)

        # Now remove any existing clicks if possible
        if len(self.events)>0:
            self.pop()

    def add_click(self,event):
        """
        This add the coordinates of an event to the list of clicks
        """
        self.clicks.append((event.xdata,event.ydata))

        verbose.report("input %i: %f,%f" % 
                       (len(self.clicks),event.xdata, event.ydata))

        # If desired plot up click
        if self.show_clicks:
            self.marks.extend(
                event.inaxes.plot([event.xdata,], [event.ydata,], 'r+') )
            self.fig.canvas.draw()

    def pop_click(self,index=-1):
        """
        This removes a click from the list of clicks.  Defaults to
        removing the last click.
        """
        self.clicks.pop(index)
        
        if self.show_clicks:
            mark = self.marks.pop(index)
            mark.remove()
            self.fig.canvas.draw()

    def pop(self,index=-1):
        """
        This removes a click and the associated event from the object.
        Defaults to removing the last click, but any index can be
        supplied.
        """
        self.pop_click(index)
        BlockingInput.pop(self,index)

    def cleanup(self):
        # clean the figure
        if self.show_clicks:
            for mark in self.marks:
                mark.remove()
            self.marks = []
            self.fig.canvas.draw()

        # Call base class to remove callbacks
        BlockingInput.cleanup(self)
        
    def __call__(self, n=1, timeout=30, show_clicks=True):
        """
        Blocking call to retrieve n coordinate pairs through mouse
        clicks.
        """
        self.show_clicks = show_clicks
        self.clicks      = []
        self.marks       = []
        BlockingInput.__call__(self,n=n,timeout=timeout)

        return self.clicks

class BlockingContourLabeler( BlockingMouseInput ):
    """
    Class that creates a callable object that uses mouse clicks on a
    figure window to place contour labels.
    """
    def __init__(self,cs):
        self.cs = cs
        BlockingMouseInput.__init__(self, fig=cs.ax.figure )

    def button1(self,event):
        """
        This will be called if an event involving a button other than
        2 or 3 occcurs.  This will add a label to a contour.
        """
        if event.inaxes == self.cs.ax:
            conmin,segmin,imin,xmin,ymin = self.cs.find_nearest_contour(
                event.x, event.y)[:5]

            paths = self.cs.collections[conmin].get_paths()
            lc = paths[segmin].vertices

            # Figure out label rotation.  This is very cludgy.
            # Ideally, there would be one method in ContourLabeler
            # that would figure out the best rotation for a label at a
            # point, but the way automatic label rotation is done is
            # quite mysterious to me and doesn't seem easy to
            # generalize to non-automatic label placement.  The method
            # used below is not very robust!  It basically looks one
            # point before and one point after label location on
            # contour and takes mean of angles of two vectors formed.
            # This produces "acceptable" results, but not nearly as
            # nice as automatic method.
            ll = lc[max(0,imin-1):imin+2] # Get points around point
            dd = np.diff(ll,axis=0)
            rotation = np.mean( np.arctan2(dd[:,1], dd[:,0]) ) * 180 / np.pi
            if rotation > 90:
                rotation = rotation -180
            if rotation < -90:
                rotation = 180 + rotation

            self.cs.add_label(xmin,ymin,rotation,conmin)

            if self.inline:
                # Get label width for breaking contours
                lw = self.cs.get_label_width(self.cs.label_levels[conmin], 
                                             self.cs.fmt, 
                                             self.cs.fslist[conmin])
                # Break contour
                new=self.cs.break_linecontour(lc,rotation,lw,imin)
                if len(new[0]):
                    paths[segmin] = path.Path(new[0])
                if len(new[1]):
                    paths.extend([path.Path(new[1])])

            self.fig.canvas.draw()
        else: # Remove event if not valid
            BlockingInput.pop(self)
            
    def button3(self,event):
        """
        This will be called if button 3 is clicked.  This will remove
        a label if not in inline mode.  Unfortunately, if one is doing
        inline labels, then there is currently no way to fix the
        broken contour - once humpty-dumpty is broken, he can't be put
        back together.  In inline mode, this does nothing.
        """
        if self.inline:
            pass
        else:
            self.cs.pop_label()
            self.cs.ax.figure.canvas.draw()

    def __call__(self,inline,n=-1,timeout=-1):
        self.inline=inline
        BlockingMouseInput.__call__(self,n=n,timeout=timeout,
                                    show_clicks=False)

class BlockingKeyMouseInput(BlockingInput):
    """
    Class that creates a callable object to retrieve a single mouse or
    keyboard click
    """
    def __init__(self, fig):
        BlockingInput.__init__(self, fig=fig, eventslist=('button_press_event','key_press_event') )

    def post_event(self):
        """
        Determines if it is a key event
        """
        assert len(self.events)>0, "No events yet"

        self.keyormouse = self.events[-1].name == 'key_press_event'

    def __call__(self, timeout=30):
        """
        Blocking call to retrieve a single mouse or key click
        Returns True if key click, False if mouse, or None if timeout
        """
        self.keyormouse = None
        BlockingInput.__call__(self,n=1,timeout=timeout)

        return self.keyormouse

