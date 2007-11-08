#------------------------------------------------------------------------------
# Copyright (c) 2005, Enthought, Inc.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
# Thanks for using Enthought open source!
#
# Author: David C. Morrill
# Original Date: 06/21/2002
#
#  Symbols defined: TraitChangeNotifyWrapper
#                   UITraitChangeNotifyWrapper
#                   NewTraitChangeNotifyWrapper
#                   StaticAnyTraitChangeNotifyWrapper
#                   StaticTraitChangeNotifyWrapper
#
#  Refactored into a separate module: 07/04/2003
#------------------------------------------------------------------------------
""" Defines the classes needed to implement and support the Traits change
notification mechanism.
"""
#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import weakref
import traceback
import sys

try:
    # Requires Python 2.4:
    from threading import local as thread_local
except:
    thread_local = lambda: {}
    
from threading \
    import Thread
    
from thread \
    import get_ident

from types \
    import MethodType

from trait_errors \
    import TraitNotificationError
    
#-------------------------------------------------------------------------------
#  Global Data:  
#-------------------------------------------------------------------------------

# The thread ID for the user interface thread
ui_thread = -1

# The handler for notifications that must be run on the UI thread
ui_handler = None

#-------------------------------------------------------------------------------
#  Sets up the user interface thread handler:  
#-------------------------------------------------------------------------------

def set_ui_handler ( handler ):
    """ Sets up the user interface thread handler.
    """
    global ui_handler, ui_thread
    
    ui_handler = handler
    ui_thread  = get_ident()

#-------------------------------------------------------------------------------
#  'NotificationExceptionHandlerState' class:  
#-------------------------------------------------------------------------------
        
class NotificationExceptionHandlerState ( object ):
    
    def __init__ ( self, handler, reraise_exceptions, locked ):
        self.handler            = handler
        self.reraise_exceptions = reraise_exceptions
        self.locked             = locked
        
#-------------------------------------------------------------------------------
#  'NotificationExceptionHandler' class:  
#-------------------------------------------------------------------------------

class NotificationExceptionHandler ( object ):
    
    def __init__ ( self ):
        self.traits_logger = None
        self.main_thread   = None
        self.thread_local  = thread_local()
        
#-- Private Methods ------------------------------------------------------------
    
    def _push_handler ( self, handler = None, reraise_exceptions = False, 
                              main = False, locked = False ):
        """ Pushes a new traits notification exception handler onto the stack,
            making it the new exception handler. Returns a
            NotificationExceptionHandlerState object describing the previous
            exception handler.
            
            Parameters
            ----------
            handler : handler
                The new exception handler, which should be a callable or 
                None. If None (the default), then the default traits 
                notification exception handler is used. If *handler* is not
                None, then it must be a callable which can accept four
                arguments: object, trait_name, old_value, new_value.
            reraise_exceptions : Boolean
                Indicates whether exceptions should be reraised after the 
                exception handler has executed. If True, exceptions will be
                re-raised after the specified handler has been executed. 
                The default value is False.
            main : Boolean
                Indicates whether the caller represents the main application
                thread. If True, then the caller's exception handler is 
                made the default handler for any other threads that are 
                created. Note that a thread can explictly set its own exception
                handler if desired. The *main* flag is provided to make it 
                easier to set a global application policy without having to 
                explicitly set it for each thread. The default value is 
                False.
            locked : Boolean
                Indicates whether further changes to the Traits notification 
                exception handler state should be allowed. If True, then 
                any subsequent calls to _push_handler() or _pop_handler() for 
                that thread will raise a TraitNotificationError. The default 
                value is False.
        """
        handlers = self._get_handlers()
        self._check_lock( handlers )
        if handler is None:
            handler = self._log_exception
        handlers.append( NotificationExceptionHandlerState( handler,
                                                  reraise_exceptions, locked ) )
        if main:
            self.main_thread = handlers
            
        return handlers[-2]
        
    def _pop_handler ( self ):
        """ Pops the traits notification exception handler stack, restoring 
            the exception handler in effect prior to the most recent 
            _push_handler() call. If the stack is empty or locked, a
            TraitNotificationError exception is raised.
            
            Note that each thread has its own independent stack. See the
            description of the _push_handler() method for more information on
            this.
        """
        handlers = self._get_handlers()
        self._check_lock( handlers )
        if len( handlers ) > 1:
            handlers.pop()
        else:
            raise TraitNotificationError( 
                      'Attempted to pop an empty traits notification exception '
                      'handler stack.' )

    def _handle_exception ( self, object, trait_name, old, new  ):
        """ Handles a traits notification exception using the handler defined
            by the topmost stack entry for the corresponding thread.
        """
        excp_class, excp = sys.exc_info()[:2]
        handler_info     = self._get_handlers()[-1]
        handler_info.handler( object, trait_name, old, new )
        if (handler_info.reraise_exceptions or
            isinstance( excp, TraitNotificationError )):
            raise excp
        
    def _get_handlers ( self ):
        """ Returns the handler stack associated with the currently executing
            thread.
        """
        thread_local = self.thread_local
        if isinstance( thread_local, dict ):
            id       = get_ident()
            handlers = thread_local.get( id )
        else:
            handlers = getattr( thread_local, 'handlers', None )
             
        if handlers is None:
            if self.main_thread is not None:
                handler = self.main_thread[-1]
            else:
                handler = NotificationExceptionHandlerState( 
                              self._log_exception, False, False )
            handlers = [ handler ]
            if isinstance( thread_local, dict ):
                thread_local[ id ] = handlers
            else:
                thread_local.handlers = handlers
            
        return handlers
            
    def _check_lock ( self, handlers ):
        """ Raises an exception if the specified handler stack is locked.
        """
        if handlers[-1].locked:
            raise TraitNotificationError( 
                      'The traits notification exception handler is locked. '
                      'No changes are allowed.' )

    #---------------------------------------------------------------------------
    #  This method defines the default notification exception handling 
    #  behavior of traits. However, it can be completely overridden by pushing
    #  a new handler using the '_push_handler' method.
    #
    #  It logs any exceptions generated in a trait notification handler.
    #---------------------------------------------------------------------------

    def _log_exception ( self, object, trait_name, old, new ):
        """ Logs any exceptions generated in a trait notification handler.
        """
        # When the stack depth is too great, the logger can't always log the 
        # message. Make sure that it goes to the console at a minimum:
        excp_class, excp = sys.exc_info()[:2]
        if ((excp_class is RuntimeError) and
            (excp.args[0] == 'maximum recursion depth exceeded')):
            sys.__stderr__.write( 'Exception occurred in traits notification '
                'handler for object: %s, trait: %s, old value: %s, '
                'new value: %s.\n%s\n' % ( object, trait_name, old, new, 
                ''.join( traceback.format_exception( *sys.exc_info() ) ) ) )
    
        logger = self.traits_logger
        if logger is None:
            import logging
            
            self.traits_logger = logger = logging.getLogger(
                                                  'enthought.traits' )
            handler = logging.StreamHandler()
            handler.setFormatter( logging.Formatter( '%(message)s' ) )
            logger.addHandler( handler )
            print ('Exception occurred in traits notification handler.\n'
                   'Please check the log file for details.')
    
        try:
            logger.exception( 
                'Exception occurred in traits notification handler for '
                'object: %s, trait: %s, old value: %s, new value: %s' %
                ( object, trait_name, old, new ) )
        except Exception:
            # Ignore anything we can't log the above way:
            pass

#-------------------------------------------------------------------------------
#  Traits global notification exception handler:  
#-------------------------------------------------------------------------------

notification_exception_handler = NotificationExceptionHandler()

push_exception_handler = notification_exception_handler._push_handler
pop_exception_handler  = notification_exception_handler._pop_handler
handle_exception       = notification_exception_handler._handle_exception
        
#-------------------------------------------------------------------------------
#  'StaticAnyTraitChangeNotifyWrapper' class:
#-------------------------------------------------------------------------------

class StaticAnyTraitChangeNotifyWrapper:

    def __init__ ( self, handler ):
         self.handler  = handler
         self.__call__ = getattr( self, 'call_%d' %
                                        handler.func_code.co_argcount )

    def equals ( self, handler ):
        return False

    def call_0 ( self, object, trait_name, old, new ):
        try:
            self.handler()
        except:
            handle_exception( object, trait_name, old, new )

    def call_1 ( self, object, trait_name, old, new ):
        try:
            self.handler( object )
        except:
            handle_exception( object, trait_name, old, new )

    def call_2 ( self, object, trait_name, old, new ):
        try:
            self.handler( object, trait_name )
        except:
            handle_exception( object, trait_name, old, new )

    def call_3 ( self, object, trait_name, old, new ):
        try:
            self.handler( object, trait_name, new )
        except:
            handle_exception( object, trait_name, old, new )

    def call_4 ( self, object, trait_name, old, new ):
        try:
            self.handler( object, trait_name, old, new )
        except:
            handle_exception( object, trait_name, old, new )

#-------------------------------------------------------------------------------
#  'StaticTraitChangeNotifyWrapper' class:
#-------------------------------------------------------------------------------

class StaticTraitChangeNotifyWrapper:

    def __init__ ( self, handler ):
        self.handler  = handler
        self.__call__ = getattr( self, 'call_%d' %
                                       handler.func_code.co_argcount )

    def equals ( self, handler ):
        return False

    def call_0 ( self, object, trait_name, old, new ):
        try:
            self.handler()
        except:
            handle_exception( object, trait_name, old, new )

    def call_1 ( self, object, trait_name, old, new ):
        try:
            self.handler( object )
        except:
            handle_exception( object, trait_name, old, new )

    def call_2 ( self, object, trait_name, old, new ):
        try:
            self.handler( object, new )
        except:
            handle_exception( object, trait_name, old, new )

    def call_3 ( self, object, trait_name, old, new ):
        try:
            self.handler( object, old, new )
        except:
            handle_exception( object, trait_name, old, new )

    def call_4 ( self, object, trait_name, old, new ):
        try:
            self.handler( object, trait_name, old, new )
        except:
            handle_exception( object, trait_name, old, new )

#-------------------------------------------------------------------------------
#  'TraitChangeNotifyWrapper' class:
#-------------------------------------------------------------------------------

class TraitChangeNotifyWrapper:

    def __init__ ( self, handler, owner ):
        func = handler
        if type( handler ) is MethodType:
            func   = handler.im_func
            object = handler.im_self
            if object is not None:
                self.object   = weakref.ref( object, self.listener_deleted )
                self.name     = handler.__name__
                self.owner    = owner
                self.__call__ = getattr( self, 'rebind_call_%d' % 
                                         (func.func_code.co_argcount - 1) )
                return
        self.name     = None
        self.handler  = handler
        self.__call__ = getattr( self, 'call_%d' % 
                                       handler.func_code.co_argcount )
                                 
    # NOTE: This method is normally the only one that needs to be overridden in 
    # a subclass to implement the subclass's dispatch mechanism: 
    def dispatch ( self, handler, *args ):
        handler( *args )

    def equals ( self, handler ):
        if handler is self:
            return True
        if (type( handler ) is MethodType) and (handler.im_self is not None):
            return ((handler.__name__ == self.name) and
                    (handler.im_self is self.object()))
        return ((self.name is None) and (handler == self.handler))

    def listener_deleted ( self, ref ):
        self.owner.remove( self )
        self.object = self.owner = None

    def dispose ( self ):
        self.object = None
        
    def call_0 ( self, object, trait_name, old, new ):
        try:
            self.dispatch( self.handler )
        except:
            handle_exception( object, trait_name, old, new )

    def call_1 ( self, object, trait_name, old, new ):
        try:
            self.dispatch( self.handler, new )
        except:
            handle_exception( object, trait_name, old, new )

    def call_2 ( self, object, trait_name, old, new ):
        try:
            self.dispatch( self.handler, trait_name, new )
        except:
            handle_exception( object, trait_name, old, new )

    def call_3 ( self, object, trait_name, old, new ):
        try:
            self.dispatch( self.handler, object, trait_name, new )
        except:
            handle_exception( object, trait_name, old, new )

    def call_4 ( self, object, trait_name, old, new ):
        try:
            self.dispatch( self.handler, object, trait_name, old, new )
        except:
            handle_exception( object, trait_name, old, new )

    def rebind_call_0 ( self, object, trait_name, old, new ):
        try:
            self.dispatch( getattr( self.object(), self.name ) )
        except:
            handle_exception( object, trait_name, old, new )

    def rebind_call_1 ( self, object, trait_name, old, new ):
        try:
            self.dispatch( getattr( self.object(), self.name ), new )
        except:
            handle_exception( object, trait_name, old, new )

    def rebind_call_2 ( self, object, trait_name, old, new ):
        try:
            self.dispatch( getattr( self.object(), self.name ), 
                           trait_name, new )
        except:
            handle_exception( object, trait_name, old, new )

    def rebind_call_3 ( self, object, trait_name, old, new ):
        try:
            self.dispatch( getattr( self.object(), self.name ), 
                           object, trait_name, new )
        except:
            handle_exception( object, trait_name, old, new )

    def rebind_call_4 ( self, object, trait_name, old, new ):
        try:
            self.dispatch( getattr( self.object(), self.name ), 
                           object, trait_name, old, new )
        except:
            handle_exception( object, trait_name, old, new )

#-------------------------------------------------------------------------------
#  'UITraitChangeNotifyUIWrapper' class:
#-------------------------------------------------------------------------------

class UITraitChangeNotifyWrapper ( TraitChangeNotifyWrapper ):

    def dispatch ( self, handler, *args ):
        if get_ident() == ui_thread:
            handler( *args )
        else:
            ui_handler( handler, *args )

#-------------------------------------------------------------------------------
#  'NewTraitChangeNotifyWrapper' class:
#-------------------------------------------------------------------------------

class NewTraitChangeNotifyWrapper ( TraitChangeNotifyWrapper ):

    def dispatch ( self, handler, *args ):
        Thread( target = handler, args = args ).start()
            
