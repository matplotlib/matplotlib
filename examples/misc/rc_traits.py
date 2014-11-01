# Here is some example code showing how to define some representative
# rc properties and construct a matplotlib artist using traits.
# matplotlib does not ship with enthought.traits, so you will need to
# install it separately.

from __future__ import print_function

import sys, os, re
import traits.api as traits
from matplotlib.cbook import is_string_like
from matplotlib.artist import Artist

doprint = True
flexible_true_trait = traits.Trait(
   True,
   { 'true':  True, 't': True, 'yes': True, 'y': True, 'on':  True, True: True,
     'false': False, 'f': False, 'no':  False, 'n': False, 'off': False, False: False
                              } )
flexible_false_trait = traits.Trait( False, flexible_true_trait )

colors = {
   'c' : '#00bfbf',
   'b' : '#0000ff',
   'g' : '#008000',
   'k' : '#000000',
   'm' : '#bf00bf',
   'r' : '#ff0000',
   'w' : '#ffffff',
   'y' : '#bfbf00',
   'gold'                 : '#FFD700',
   'peachpuff'            : '#FFDAB9',
   'navajowhite'          : '#FFDEAD',
   }

def hex2color(s):
   "Convert hex string (like html uses, e.g., #efefef) to a r,g,b tuple"
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
       return 'r,g,b,a = (%1.2f, %1.2f, %1.2f, %1.2f)'%\
              (self.r, self.g, self.b, self.a)

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
             hex_to_rgba, tuple_to_rgba)

def file_exists(ob, name, val):
   fh = file(val, 'r')
   return val

def path_exists(ob, name, val):
   os.path.exists(val)
linestyles  = ('-', '--', '-.', ':', 'steps', 'None')
TICKLEFT, TICKRIGHT, TICKUP, TICKDOWN = range(4)
linemarkers = (None, '.', ',', 'o', '^', 'v', '<', '>', 's',
                 '+', 'x', 'd', 'D', '|', '_', 'h', 'H',
                 'p', '1', '2', '3', '4',
                 TICKLEFT,
                 TICKRIGHT,
                 TICKUP,
                 TICKDOWN,
                 'None'
              )

class LineRC(traits.HasTraits):
   linewidth       = traits.Float(0.5)
   linestyle       = traits.Trait(*linestyles)
   color           = Color
   marker          = traits.Trait(*linemarkers)
   markerfacecolor = Color
   markeredgecolor = Color
   markeredgewidth = traits.Float(0.5)
   markersize      = traits.Float(6)
   antialiased     = flexible_true_trait
   data_clipping   = flexible_false_trait

class PatchRC(traits.HasTraits):
   linewidth       = traits.Float(1.0)
   facecolor = Color
   edgecolor = Color
   antialiased     = flexible_true_trait

timezones = 'UTC', 'US/Central', 'ES/Eastern' # fixme: and many more
backends = ('GTKAgg', 'Cairo', 'GDK', 'GTK', 'Agg',
           'GTKCairo', 'PS', 'SVG', 'Template', 'TkAgg',
           'WX')

class RC(traits.HasTraits):
   backend = traits.Trait(*backends)
   interactive  = flexible_false_trait
   toolbar      = traits.Trait('toolbar2', 'classic', None)
   timezone     = traits.Trait(*timezones)
   lines        = traits.Trait(LineRC())
   patch        = traits.Trait(PatchRC())

rc = RC()
rc.lines.color = 'r'
if doprint:
   print('RC')
   rc.print_traits()
   print('RC lines')
   rc.lines.print_traits()
   print('RC patches')
   rc.patch.print_traits()


class Patch(Artist, traits.HasTraits):
   linewidth = traits.Float(0.5)
   facecolor = Color
   fc = facecolor
   edgecolor = Color
   fill = flexible_true_trait
   def __init__(self,
                edgecolor=None,
                facecolor=None,
                linewidth=None,
                antialiased = None,
                fill=1,
                **kwargs
                ):
       Artist.__init__(self)

       if edgecolor is None: edgecolor = rc.patch.edgecolor
       if facecolor is None: facecolor = rc.patch.facecolor
       if linewidth is None: linewidth = rc.patch.linewidth
       if antialiased is None: antialiased = rc.patch.antialiased

       self.edgecolor = edgecolor
       self.facecolor = facecolor
       self.linewidth = linewidth
       self.antialiased = antialiased
       self.fill = fill


p = Patch()
p.facecolor = '#bfbf00'
p.edgecolor = 'gold'
p.facecolor = (1,.5,.5,.25)
p.facecolor = 0.25
p.fill = 'f'
print('p.facecolor', type(p.facecolor), p.facecolor)
print('p.fill', type(p.fill), p.fill)
if p.fill_: print('fill')
else: print('no fill')
if doprint:
   print()
   print('Patch')
   p.print_traits()
