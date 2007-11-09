#-----------------------------------------------------------------------------
#
#  Copyright (c) 2006 by Enthought, Inc.
#  All rights reserved.
#
#-----------------------------------------------------------------------------

"""
Provides a framework that assembles Traits UI Views at run time,
when the view is requested, rather than at the time a class is written.

This capability is particularly useful when the object being 'viewed' with a
Traits UI is part of a plug-in application -- such as Envisage.  In general,
this capability allows:

1. The GUI for an object can be extendable by contributions
   other than from the original code writer.
2. The view can be dynamic in that the elements it is composed
   of can change each time it is requested.
3. Registration of a handler can be associated with the view contributions.

Either the original object writer, or a contributor, can use this framework to
declare one or more dynamic views that are composed of sub-elements that only
need to exist at the time the view is requested.

Users of this framework create a dynamic view by registering a DynamicView
declaration.  That declaration includes a name that forms the basis for the
metadata attributes that are used to identify and order the desired view
sub-elements into the view's composition.  In addition, the declaration
includes any data to be passed into the constructor of the dynamic view and
the id that should be used to persist the user's customization of the view.

Additionally, this framework allows sub-elements themselves to also be
dynamically composed of further sub-elements.

For example, a dynamic view could be composed of two sub-elements:

1. The first is a dynamically composed HFlow, which represents a toolbar
   that can be extended through contributions of toolbar buttons.
2. The second could be a dynamic tabset where each page is also a
   contribution.

Programmers include dynamic sub-elements within their dynamic views by
contributing a DynamicViewSubElement into that view.  When the framework comes
across this contribution while building the view, it replaces that
DynamicViewSubElement with a fully initialized Traits ViewSubElement composed
in a manner similar to how the elements of the View itself were composed.

Each contribution to a dynamic view or sub-element must be an instance of a
Traits ViewSubElement and must have associated metadata like the following for
each dynamic view or sub-element it will display in:

_<dynamic name>_order : A float value.
    The framework uses only ViewSubElements with this metadata
    instantiated when building the dynamic view or sub-element with
    the specified name.  The elements are sorted by ascending
    order of this value using the standard list sort function.
_<dynamic name>_priority : A float value.
    The framework resolves any overloading of an order value by
    picking the first element encountered that has the highest
    priority value. The other elements with the same view order are
    not displayed at all.

In addition, dynamic view contributions can also provide a 'handler', which
behaves like a normal Traits Handler.  That is, it can contain methods that
are called when model values change and can access  the Traits UIInfo
object representing the actual UI instances.  To provide a handler, append the
following metadata to your view sub-element:

_<dynamic_name>_handler : A HasTraits instance.
    The framework will connect listeners to call the handler methods as
    part of the handler for the dynamic view.


"""


# Standard library imports.
import logging

# Enthought library imports.
from enthought.traits.ui.delegating_handler import \
    DelegatingHandler

# Local imports.
from has_traits import HasTraits
from traits import Any, Bool, Dict, Instance, Str
from ui.api import View, ViewSubElement, ViewElement


# Setup a logger for this module.
logger = logging.getLogger(__name__)


class DynamicViewSubElement(ViewSubElement):
    """
    Declares a dynamic sub-element of a dynamic view.

    """

    ##########################################################################
    # Traits
    ##########################################################################

    #### public 'DynamicViewSubElement' interface ############################

    # Keyword arguments passed in during construction of the actual
    # ViewSubElement instance.
    keywords = Dict

    # The class of the actual ViewSubElement we are dynamically creating.
    klass = Any # FIXME: Should be the 'Class' trait but I couldn't get that to work.

    # The name of this dynamic sub-element.  This controls the metadata
    # names identifying the sub-elements that compose this element.
    name = Str


class DynamicView(HasTraits):
    """
    Declares a dynamic view.

    """

    ##########################################################################
    # Traits
    ##########################################################################

    #### public 'DynamicView' interface ######################################

    # The ID of the view.  This is the ID that the view's preferences will be
    # saved under.
    id = Str

    # The name of the view.  This is the name that should be requested when
    # calling edit_traits() or configure_traits().
    name = Str

    # Keyword arguments passed in during construction of the actual view
    # instance.
    keywords = Dict

    # Indicates whether this view should be the default traits view for objects
    # it is contributed to.
    use_as_default = Bool(False)


class HasDynamicViews(HasTraits):
    """
    Provides of a framework that builds Traits UI Views at run time,
    when the view is requested, rather than at the time a class is written.

    """

    ##########################################################################
    # Traits
    ##########################################################################

    #### protected 'HasDynamicViews' interface ###############################

    # The registry of dynamic views.  The key is the view name and the value
    # is the declaration of the dynamic view.
    _dynamic_view_registry = Dict(Str, Instance(DynamicView))


    ##########################################################################
    # 'HasTraits' interface
    ##########################################################################

    #### public interface ####################################################

    def trait_view(self, name=None, view_element=None):
        """
        Gets or sets a ViewElement associated with an object's class.

        Extended here to build dynamic views and sub-elements.

        """

        result = None

        # If a view element was passed instead of a name or None, do not handle
        # this as a request for a dynamic view and let the standard Traits
        # trait_view method be called with it.  Otherwise, compose the dynamic
        # view here.
        if not isinstance(name, ViewElement):

            # If this is a request for the default view, see if one of our
            # dynamic views should be the default view.
            if view_element is None and (name is None or len(name) < 1):
                for dname, declaration in self._dynamic_view_registry.items():
                    if declaration.use_as_default:
                        result = self._compose_dynamic_view(dname)
                        break

            # Otherwise, handle if this is a request for a dynamic view.
            elif view_element is None and name in self._dynamic_view_registry:
                result = self._compose_dynamic_view(name)

        # If we haven't created a dynamic view so far, then do the standard
        # traits thing to retrieve the UI element.
        if result is None:
            result = super(HasDynamicViews, self).trait_view(name,
                view_element)

        return result


    ##########################################################################
    # 'HasDynamicViews' interface
    ##########################################################################

    #### public interface ####################################################

    def declare_dynamic_view(self, declaration):
        """
        A convenience method to add a new dynamic view declaration to this
        instance.

        """

        self._dynamic_view_registry[declaration.name] = declaration


    #### protected interface #################################################

    def _build_dynamic_sub_element(self, definition, sub_elements):
        """
        Returns the fully composed ViewSubElement from the sub-element
        contributions to the dynamic sub-element identified by the definition.

        """

        logger.debug('\tBuilding dynamic sub-element [%s] with elements [%s]',
            definition.name, sub_elements)

        result = definition.klass(
            *sub_elements,
            **definition.keywords
            )

        return result


    def _build_dynamic_view(self, declaration, sub_elements, handler):
        """
        Returns a Traits View representing the specified dynamic view composed
        out of the provided view sub-elements.

        Implemented as a separate method to allow implementors to override
        the way in which the instantiated view is configured.

        """

        logger.debug('\tBuilding dynamic view [%s] with elements [%s]',
            declaration.name, sub_elements)

        result = View(
            # The view id allows the user's customization of this view, if any,
            # to be persisted when the view is closed and then that persisted
            # configuration to be applied when the view is next shown.
            id = declaration.id,

            # Include the specified handler
            handler = handler,

            # Build the view out of the sub-elements
            *sub_elements,

            # Include the declaration's keywords.
            **declaration.keywords
            )

        return result


    def _compose_dynamic_sub_element(self, definition):
        """
        Returns a dynamic UI element composed from its contributed parts.

        """

        logger.debug('Composing dynamic sub-element named [%s] for [%s]',
            definition.name, self)

        # Retrieve the set of elements that make up this sub-element.
        elements = self._get_dynamic_elements(definition.name)

        # Build the sub-element.
        return self._build_dynamic_sub_element(definition, elements)


    def _compose_dynamic_view(self, name):
        """
        Returns a dynamic view composed from its contributed parts.

        """

        logger.debug('Composing dynamic view [%s] for [%s]', name, self)

        # Retrieve the declaration of this dynamic view
        declaration = self._dynamic_view_registry[name]

        # Retrieve the set of elements that make up the view
        elements = self._get_dynamic_elements(declaration.name)

        # Build a handler that delegates to the contribution handlers if any
        # exist.
        handler = None
        handlers = self._get_dynamic_handlers(declaration.name, elements)
        if len(handlers) > 0:
            handler = DelegatingHandler(sub_handlers = handlers)

        # Build the view.
        return self._build_dynamic_view(declaration, elements, handler)


    def _get_dynamic_elements(self, name):
        """
        Returns a list of the current elements meant to go into the composition
        of a dynamic view or sublement with the specified name.

        """

        # Determine the metadata names used to find the sub-elements included
        # within this dynamic element.
        name = name.replace(' ', '_')
        order_trait_name = '_%s_order' % name
        priority_trait_name = '_%s_priority' % name

        # Now find all of the current sub-elements that we will use when
        # composing our element.
        all_elements = [ self.trait_view(g) for g in \
            self.trait_views(klass=ViewSubElement) ]
        elements = [ e for e in all_elements \
            if hasattr(e, order_trait_name) and \
            getattr(e, order_trait_name) is not None ]

        # Filter out any overridden elements.  This means taking out the
        # element with the lower priority whenever two elements have the
        # same order value.
        filtered = {}
        for e in elements:
            order = getattr(e, order_trait_name)
            priority = getattr(e, priority_trait_name) or 0
            current = filtered.setdefault(order, e)
            if current is not e:
                current_priority = getattr(current, priority_trait_name)
                if current_priority < priority:
                    filtered[order] = e

        # Sort the contributed elements by their display ordering values.
        ordering = filtered.keys()
        ordering.sort()
        elements = [filtered[order] for order in ordering]

        # Replace any dynamic sub-element with their full composition.
        # NOTE: We can't do this in the override of 'trait_view' because
        # then we get into infinite loops when a dynamic view subelement is
        # found as a child.
        for i in range(len(elements)):
            if isinstance(elements[i], DynamicViewSubElement):
                e = elements.pop(i)
                composed = self._compose_dynamic_sub_element(e)
                elements.insert(i, composed)

        return elements


    def _get_dynamic_handlers(self, name, elements):
        """
        Return a list of the handlers associated with the current elements
        meant to go into the dynamic view of the specified name.

        """

        # Determine the metadata name used to declare a handler
        name = name.replace(' ', '_')
        handler_name = '_%s_handler' % name

        handlers = ([ getattr(e, handler_name) for e in elements
            if hasattr(e, handler_name) and
            getattr(e, handler_name) is not None
            ])
        logger.debug('\tFound sub-handlers: %s', handlers)

        return handlers


#### EOF #####################################################################

