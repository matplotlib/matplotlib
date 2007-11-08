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
# Date: 10/14/2004
#             
# Symbols defined:
#
#------------------------------------------------------------------------------
""" Defines common traits used within the traits.ui package.
"""
#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from enthought.traits.api \
    import Trait, TraitPrefixList, Delegate, Str, Instance, List, Enum, Any

#-------------------------------------------------------------------------------
#  Trait definitions:
#-------------------------------------------------------------------------------

# Styles for user interface elements
style_trait = Trait( 'simple',
                     TraitPrefixList( 'simple', 'custom', 'text', 'readonly' ),
                     cols = 4 )
                     
# Trait for the default object being edited                     
object_trait = Str( 'object' )                     

# The default dock style to use
dock_style_trait = Enum( 'fixed', 'horizontal', 'vertical', 'tab',
                         desc = "the default docking style to use" )
                         
# The default notebook tab image to use                      
image_trait = Instance( 'enthought.pyface.image_resource.ImageResource',
                        desc = 'the image to be displayed on notebook tabs' )
                     
# The category of elements dragged out of the view
export_trait = Str( desc = 'the category of elements dragged out of the view' )

# Delegate a trait value to the object's **container** trait                  
container_delegate = Delegate( 'container' )

# An identifier for the external help context
help_id_trait = Str( desc = "the external help context identifier" )                     

# A button to add to a view
a_button = Trait( '', Str, Instance( 'enthought.traits.ui.menu.Action' ) )
# The set of buttons to add to the view
buttons_trait = List( a_button,
                      desc = 'the action buttons to add to the bottom of '
                             'the view' )

# View trait specified by name or instance:
AView = Any
#AView = Trait( '', Str, Instance( 'enthought.traits.ui.View' ) )

#-------------------------------------------------------------------------------
#  Other definitions:
#-------------------------------------------------------------------------------

# Types that represent sequences
SequenceTypes = ( tuple, list )
