#-------------------------------------------------------------------------------
#
#  Trait definition for a 'null' (i.e. no UI)-based font.
#
#  Written by: David C. Morrill
#
#  Date: 02/14/2005
#
#  (c) Copyright 2005 by Enthought, Inc.
#
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from matplotlib.enthought.traits    import Trait, TraitHandler, TraitError
from matplotlib.enthought.traits.ui import FontEditor

#-------------------------------------------------------------------------------
#  Convert a string into a valid 'wxFont' object (if possible):
#-------------------------------------------------------------------------------

font_families = [
    'default',
    'decorative',
    'roman',
    'script',
    'swiss',
    'modern'
]

font_styles = [
    'slant',
    'italic'
]

font_weights = [
    'light',
    'bold'
]

font_noise = [ 'pt', 'point', 'family' ]

#-------------------------------------------------------------------------------
#  'TraitFont' class'
#-------------------------------------------------------------------------------

class TraitFont ( TraitHandler ):
    
    #---------------------------------------------------------------------------
    #  Validates that the value is a valid font:
    #---------------------------------------------------------------------------
    
    def validate ( self, object, name, value ):
        """ Validates that the value is a valid font.
        """
        try:
            point_size = family = style = weight = underline = ''
            facename   = []
            for word in value.split():
                lword = word.lower()
                if lword in font_families:
                    family = ' ' + lword
                elif lword in font_styles:
                    style = ' ' + lword
                elif lword in font_weights:
                    weight = ' ' + lword
                elif lword == 'underline':
                    underline = ' ' + lword
                elif lword not in font_noise:
                    try:
                        int( lword )
                        point_size = lword + ' pt'
                    except:
                        facename.append( word )
            return ('%s%s%s%s%s%s' % ( point_size, family, style, weight, 
                    underline, ' '.join( facename ) )).strip()
        except:
            pass
        raise TraitError, ( object, name, 'a font descriptor string',
                            repr( value ) )

    def info ( self ):                              
        return ( "a string describing a font (e.g. '12 pt bold italic "
                 "swiss family Arial' or 'default 12')" )

#-------------------------------------------------------------------------------
#  Define a 'null' specific font trait:
#-------------------------------------------------------------------------------

fh       = TraitFont()
NullFont = Trait( fh.validate( None, None, 'Arial 10' ), fh, 
                  editor = FontEditor )
    
