"""
Install instructions for traits 2.0

   # blow away old enthought
   rm -rf ~/dev/lib/python2.4/site-packages/enthought.*

   # get easy_install, if necessary
   wget  http://peak.telecommunity.com/dist/ez_setup.py
   sudo python ez_setup.py

   sudo easy_install -f http://code.enthought.com/enstaller/eggs/source/unstable/ "enthought.etsconfig <3.0a" "enthought.util <3.0a" "enthought.debug <3.0a"

   svn co https://svn.enthought.com/svn/enthought/branches/enthought.traits_2.0 enthought_traits

   cd enthought_traits/
   sudo python setup.py install 


"""
# Here is some example code showing how to define some representative
# rc properties and construct a matplotlib artist using traits.
# Because matplotlib ships with enthought traits already, you can run
# this script with just matplotlib.  Unfortunately, we do not ship the
# ex UI component so you can't test that part.  I'm a bit of a traits
# newbie so there are probably better ways to do what I have done
# below.

import sys, os, re
import enthought.traits.api as traits
from matplotlib.cbook import is_string_like
from matplotlib import colors as mcolors
import numpy as npy

doprint = True
FlexibleTrueTrait = traits.Trait(
   True,
   { 'true':  True, 't': True, 'yes': True, 'y': True, 'on':  True, True: True,
     'false': False, 'f': False, 'no':  False, 'n': False, 'off': False, False: False
                              } )
FlexibleFalseTrait = traits.Trait( False, FlexibleTrueTrait )

colors = mcolors.cnames

def hex2color(s):
   "Convert hex string (like html uses, eg, #efefef) to a r,g,b tuple"
   return tuple([int(n, 16)/255.0 for n in (s[1:3], s[3:5], s[5:7])])

class RGBA(traits.HasTraits):
   # r,g,b,a in the range 0-1 with default color 0,0,0,1 (black)
   r = traits.Range(0., 1., 0.)
   g = traits.Range(0., 1., 0.)
   b = traits.Range(0., 1., 0.)
   a = traits.Range(0., 1., 1.)
   def __init__(self, r=0., g=0., b=0., a=1.):
       self.r = r
       self.g = g
       self.b = b
       self.a = a

   def __repr__(self):
       return '(%1.2f, %1.2f, %1.2f, %1.2f)'%(self.r, self.g, self.b, self.a)

def tuple_to_rgba(ob, name, val):
   tup = [float(x) for x in val]
   if len(tup)==3:
       r,g,b = tup
       return RGBA(r,g,b)
   elif len(tup)==4:
       r,g,b,a = tup
       return RGBA(r,g,b,a)
   else:
       raise ValueError
tuple_to_rgba.info = 'a RGB or RGBA tuple of floats'

def hex_to_rgba(ob, name, val):
   rgx = re.compile('^#[0-9A-Fa-f]{6}$')

   if not is_string_like(val):
       raise TypeError
   if rgx.match(val) is None:
       raise ValueError
   r,g,b = hex2color(val)
   return RGBA(r,g,b,1.0)
hex_to_rgba.info = 'a hex color string'

def colorname_to_rgba(ob, name, val):
   hex = colors[val.lower()]
   r,g,b =  hex2color(hex)
   return RGBA(r,g,b,1.0)
colorname_to_rgba.info = 'a named color'

def float_to_rgba(ob, name, val):
   val = float(val)
   return RGBA(val, val, val, 1.)
float_to_rgba.info = 'a grayscale intensity'



Color = traits.Trait(RGBA(), float_to_rgba, colorname_to_rgba, RGBA,
             hex_to_rgba, tuple_to_rgba, None)

def file_exists(ob, name, val):
   fh = file(val, 'r')
   return val

def path_exists(ob, name, val):
   os.path.exists(val)
linestyles  = ('-', '--', '-.', ':', 'steps')
TICKLEFT, TICKRIGHT, TICKUP, TICKDOWN = range(4)
linemarkers = (None, '.', ',', 'o', '^', 'v', '<', '>', 's',
                 '+', 'x', 'd', 'D', '|', '_', 'h', 'H',
                 'p', '1', '2', '3', '4',
                 TICKLEFT,
                 TICKRIGHT,
                 TICKUP,
                 TICKDOWN,
                 )
              

Alpha          = traits.Float(1.0)
Linewidth       = traits.Float(0.5)
Linestyle       = traits.Trait(*linestyles)
Color           = Color
Marker          = traits.Trait(*linemarkers)
Markersize      = traits.Float(6)
Antialiased     = FlexibleTrueTrait
Alpha           = traits.Range(0., 1., 0.)
Interval        = traits.Array('d', (2,), npy.array([0.0, 1.0], npy.float_))
Affine          = traits.Array('d', (3,3),
                               npy.array([[1,0,0],[0,1,0],[0,0,1]], npy.float_))
Verts          = traits.Array('d', value=npy.array([[0,0],[0,0]], npy.float_))
Codes          = traits.Array('b', value=npy.array([0,0], dtype=npy.uint8))
PathData       = traits.Tuple(Codes, Verts)

