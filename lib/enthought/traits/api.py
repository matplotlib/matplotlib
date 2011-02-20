#------------------------------------------------------------------------------
#
#  Copyright (c) 2005, Enthought, Inc.
#  All rights reserved.
#
#  Written by: David C. Morrill
#
#  Date: 12/06/2005
#
#------------------------------------------------------------------------------


""" Pseudo-package for all of the core symbols from Traits and TraitsUI.
Use this module for importing Traits names into your namespace. For example::

    from enthought.traits.api import HasTraits

"""

from enthought.traits.version import version, version as __version__


from info_traits \
    import __doc__

from trait_base \
    import Undefined, Missing, Self

from trait_errors \
    import TraitError, TraitNotificationError, DelegationError

from trait_notifiers \
   import push_exception_handler, pop_exception_handler, \
          TraitChangeNotifyWrapper

from category \
    import Category

from trait_db \
    import tdb

from traits \
    import Event, List, Dict, Tuple, Range, Constant, CTrait, Trait, Delegate, \
           Property, Expression, Button, ToolbarButton, PythonValue, Any, Int, \
           Long, Float, Str, Unicode, Complex, Bool, CInt, CLong, CFloat, \
           CStr, CUnicode, WeakRef

from traits \
    import CComplex, CBool, false, true, Regex, String, Password, File, \
           Directory, Function, Method, Class, Instance, Module, Type, This, \
           self, Either, Python, Disallow, ReadOnly, undefined, missing, ListInt

from traits \
    import ListFloat, ListStr, ListUnicode, ListComplex, ListBool, \
           ListFunction, ListMethod, ListClass, ListInstance, ListThis, \
           DictStrAny, DictStrStr, DictStrInt, DictStrLong, DictStrFloat

from traits \
    import DictStrBool, DictStrList, TraitFactory, Callable, Array, CArray, \
           Enum, Code, HTML, Default, Color, RGBColor, Font

from has_traits \
    import method, HasTraits, HasStrictTraits, HasPrivateTraits, \
           SingletonHasTraits, SingletonHasStrictTraits, \
           SingletonHasPrivateTraits, MetaHasTraits, Vetoable, VetoableEvent, \
           traits_super

from trait_handlers \
    import TraitHandler, TraitRange, TraitString, TraitType, TraitCastType, \
           TraitInstance, ThisClass, TraitClass, TraitFunction, TraitEnum, \
           TraitPrefixList, TraitMap, TraitPrefixMap, TraitCompound, \
           TraitList, TraitListEvent, TraitDict, TraitDictEvent, TraitTuple

from traits \
    import UIDebugger

###################
# ui imports
if False:

    from ui.handler \
        import Handler, ViewHandler, default_handler

    from ui.view \
        import View

    from ui.group \
        import Group, HGroup, VGroup, VGrid, HFlow, VFlow, HSplit, VSplit, Tabbed

    from ui.ui \
        import UI

    from ui.ui_info \
        import UIInfo

    from ui.help \
        import on_help_call

    from ui.include \
        import Include

    from ui.item \
        import Item, Label, Heading, Spring, spring

    from ui.editor_factory \
        import EditorFactory

    from ui.editor \
        import Editor

    from ui.toolkit \
        import toolkit

    from ui.undo \
        import UndoHistory, AbstractUndoItem, UndoItem, ListUndoItem, \
               UndoHistoryUndoItem

    from ui.view_element \
        import ViewElement, ViewSubElement

    from ui.help_template \
        import help_template

    from ui.message \
        import message, error

    from ui.tree_node \
        import TreeNode, ObjectTreeNode, TreeNodeObject, MultiTreeNode

    from ui.editors \
        import ArrayEditor, BooleanEditor, ButtonEditor, CheckListEditor, \
               CodeEditor, ColorEditor, RGBColorEditor, \
               CompoundEditor, DirectoryEditor, EnumEditor, FileEditor, \
               FontEditor, ImageEnumEditor, InstanceEditor, \
               ListEditor, RangeEditor, TextEditor, TreeEditor, \
               TableEditor, TupleEditor, DropEditor, DNDEditor, CustomEditor

    from ui.editors \
        import ColorTrait, RGBColorTrait, \
               FontTrait, SetEditor, HTMLEditor, KeyBindingEditor, \
               ShellEditor, TitleEditor, ValueEditor, NullEditor


import ui.view_elements

#-------------------------------------------------------------------------------
#  Patch the main traits module with the correct definition for the ViewElements
#  class:
#-------------------------------------------------------------------------------

import has_traits as has_traits
has_traits.ViewElements = ui.view_elements.ViewElements

#-------------------------------------------------------------------------------
#  Patch the main traits module with the correct definition for the ViewElement
#  and ViewSubElement class:
#-------------------------------------------------------------------------------

has_traits.ViewElement = ui.view_element.ViewElement
