r"""

OVERVIEW 

  mathtext is a module for parsing TeX expressions and drawing them
  into a matplotlib.ft2font image buffer.  You can draw from this
  buffer into your backend.

  A large set of the TeX symbols are provided (see below).
  Subscripting and superscripting are supported, as well as the
  over/under style of subscripting with \sum, \int, etc.

  The module uses pyparsing to parse the TeX expression, an so can
  handle fairly complex TeX expressions Eg, the following renders
  correctly

  s = r'$\cal{R}\prod_{i=\alpha\cal{B}}^\infty a_i\rm{sin}(2 \pi f x_i)$'

  The fonts \cal, \rm, \it, and \tt are allowed.

  The following accents are provided: \hat, \breve, \grave, \bar,
  \acute, \tilde, \vec, \dot, \ddot.  All of them have the same
  syntax, eg to make an overbar you do \bar{o} or to make an o umlaut
  you do \ddot{o}.  The shortcuts are also provided, eg: \"o \'e \`e
  \~n \.x \^y

  The spacing elements \ , \/ and \hspace{num} are provided.  \/
  inserts a small space, and \hspace{num} inserts a fraction of the
  current fontsize.  Eg, if num=0.5 and the fontsize is 12.0,
  hspace{0.5} inserts 6 points of space


  
  If you find TeX expressions that don't parse or render properly,
  please email me, but please check KNOWN ISSUES below first.

REQUIREMENTS

  mathtext requires matplotlib.ft2font.  Set BUILD_FT2FONT=True in
  setup.py.  See BACKENDS below for a summary of availability by
  backend.

LICENSING:

  The computer modern fonts this package uses are part of the BaKoMa
  fonts, which are (in my understanding) free for noncommercial use.
  For commercial use, please consult the licenses in fonts/ttf and the
  author Basil K. Malyshev - see also
  http://www.mozilla.org/projects/mathml/fonts/encoding/license-bakoma.txt
  and the file BaKoMa-CM.Fonts in the matplotlib fonts dir.

  Note that all the code in this module is distributed under the
  matplotlib license, and a truly free implementation of mathtext for
  either freetype or ps would simply require deriving another concrete
  implementation from the Fonts class defined in this module which
  used free fonts.

USAGE:

  See http://matplotlib.sourceforge.net/tutorial.html#mathtext for a
  tutorial introduction.
  
  Any text element (xlabel, ylabel, title, text, etc) can use TeX
  markup, as in

    xlabel(r'$\Delta_i$')
           ^
        use raw strings

  The $ symbols must be the first and last symbols in the string.  Eg,
  you cannot do 

    r'My label $x_i$'.  

  but you can change fonts, as in 

    r'$\rm{My label} x_i$' 

  to achieve the same effect.

  A large set of the TeX symbols are provided.  Subscripting and
  superscripting are supported, as well as the over/under style of
  subscripting with \sum, \int, etc.


  Allowed TeX symbols:

  \/ \Delta \Downarrow \Gamma \Im \LEFTangle \LEFTbrace \LEFTbracket
  \LEFTparen \Lambda \Leftarrow \Leftbrace \Leftbracket \Leftparen
  \Leftrightarrow \Omega \P \Phi \Pi \Psi \RIGHTangle \RIGHTbrace
  \RIGHTbracket \RIGHTparen \Re \Rightarrow \Rightbrace \Rightbracket
  \Rightparen \S \SQRT \Sigma \Sqrt \Theta \Uparrow \Updownarrow
  \Upsilon \Vert \Xi \aleph \alpha \approx \angstrom \ast \asymp
  \backslash \beta \bigcap \bigcirc \bigcup \bigodot \bigoplus
  \bigotimes \bigtriangledown \bigtriangleup \biguplus \bigvee
  \bigwedge \bot \bullet \cap \cdot \chi \circ \clubsuit \coprod \cup
  \dag \dashv \ddag \delta \diamond \diamondsuit \div \downarrow \ell
  \emptyset \epsilon \equiv \eta \exists \flat \forall \frown \gamma
  \geq \gg \heartsuit \hspace \imath \in \infty \int \iota \jmath
  \kappa \lambda \langle \lbrace \lceil \leftangle \leftarrow
  \leftbrace \leftbracket \leftharpoondown \leftharpoonup \leftparen
  \leftrightarrow \leq \lfloor \ll \mid \mp \mu \nabla \natural
  \nearrow \neg \ni \nu \nwarrow \odot \oint \omega \ominus \oplus
  \oslash \otimes \phi \pi \pm \prec \preceq \prime \prod \propto \psi
  \rangle \rbrace \rceil \rfloor \rho \rightangle \rightarrow
  \rightbrace \rightbracket \rightharpoondown \rightharpoonup
  \rightparen \searrow \sharp \sigma \sim \simeq \slash \smile
  \spadesuit \sqcap \sqcup \sqrt \sqsubseteq \sqsupseteq \subset
  \subseteq \succ \succeq \sum \supset \supseteq \swarrow \tau \theta
  \times \top \triangleleft \triangleright \uparrow \updownarrow
  \uplus \upsilon \varepsilon \varphi \varphi \varrho \varsigma
  \vartheta \vdash \vee \vert \wedge \wp \wr \xi \zeta

  
BACKENDS

  mathtext currently works with GTK, Agg, GTKAgg, TkAgg and WxAgg and
  PS, though only horizontal and vertical rotations are supported.  If
  David incorporates ft2font into paint, it will be easy to add to
  Paint.  WX can follow the lead of GTK and use pixels API calls to
  use mathtext.

  mathtext now embeds the TrueType computer modern fonts into the PS
  file, so what you see on the screen should be what you get on paper.

  Backends which don't support mathtext will just render the TeX
  string as a literal.  Stay tuned.


KNOWN ISSUES:

 - FIXED in 0.54 - some hackish ways I deal with a strange offset in cmex10
 - nested subscripts, eg, x_i_j not working; but you can do x_{i_j}
 - nesting fonts changes in sub/superscript groups not parsing
 - I would also like to add a few more layout commands, like \frac.

Author    : John Hunter <jdhunter@ace.bsd.uchicago.edu>
Copyright : John Hunter (2004)
License   : matplotlib license (PSF compatible)
 
"""
from __future__ import division
import os, sys
from cStringIO import StringIO

from matplotlib import verbose
from matplotlib.pyparsing import Literal, Word, OneOrMore, ZeroOrMore, \
     Combine, Group, Optional, Forward, NotAny, alphas, nums, alphanums, \
     StringStart, StringEnd, ParseException

from matplotlib.afm import AFM
from matplotlib.cbook import enumerate, iterable, Bunch
from matplotlib.ft2font import FT2Font
from matplotlib._mathtext_data import latex_to_bakoma
from matplotlib.numerix import absolute
from matplotlib import get_data_path


bakoma_fonts = []

# symbolx that have the sub and superscripts over/under 
overunder = { r'\sum'    : 1,
              r'\int'    : 1,
              r'\prod'   : 1,
              r'\coprod' : 1,
              }
# a character over another character
charOverChars = {
    # The first 2 entires in the tuple are (font, char, sizescale) for
    # the two symbols under and over.  The third entry is the space
    # between the two symbols in points
    r'\angstrom' : (  ('rm', 'A', 1.0), (None, '\circ', 0.5), 0.0 ),
    }

class Fonts:
    """
    An abstract base class for fonts that want to render mathtext

    The class must be able to take symbol keys and font file names and
    return the character metrics as well as do the drawing
    """

    def get_metrics(self, font, sym, fontsize, dpi):
        """
        font is one of tt, it, rm, cal or None

        sym is a single symbol(alphanum, punct) or a special symbol
        like \sigma.

        fontsize is in points
        
        Return object has attributes - see
        http://www.freetype.org/freetype2/docs/tutorial/step2.html for
        a pictoral representation of these attributes
        
          advance
          height
          width
          xmin, xmax, ymin, ymax  - the ink rectangle of the glyph
          """
        raise NotImplementedError('Derived must override')

    def set_canvas_size(self, w, h):
        'Dimension the drawing canvas; may be a noop'
        self.width, self.height = w, h

    def render(self, ox, oy, font, sym, fontsize, dpi):
        pass


class DummyFonts:
    'dummy class for debugging parser'
    def get_metrics(self, font, sym, fontsize, dpi):

        metrics = Bunch(
            advance  = 0,
            height   = 0,
            width    = 0,
            xmin = 0,
            xmax = 0,
            ymin = 0,
            ymax = 0,
            )
        return metrics

class BakomaTrueTypeFonts(Fonts):
    """
    Use the Bakoma true type fonts for rendering
    """
    fnames = ('cmmi10', 'cmsy10', 'cmex10',
              'cmtt10', 'cmr10')
    # allocate a new set of fonts
    basepath = get_data_path()
    
    fontmap = { 'cal' : 'cmsy10',
                'rm'  : 'cmr10',
                'tt'  : 'cmtt10',
                'it'  : 'cmmi10',
                None  : 'cmmi10',
                }

    def __init__(self, useSVG=False):
        self.glyphd = {}
        self.fonts = dict(
            [ (name, FT2Font(os.path.join(self.basepath, name) + '.ttf'))
              for name in self.fnames])

        self.charmaps = dict(
            [ (name, self.fonts[name].get_charmap()) for name in self.fnames])
        
        for font in self.fonts.values():
            font.clear()
        if useSVG:
            self.svg_glyphs=[]  # a list of "glyphs" we need to render this thing in SVG
        else: pass
        self.usingSVG = useSVG
            
    def get_metrics(self, font, sym, fontsize, dpi):
        cmfont, metrics, glyph, offset  = \
                self._get_info(font, sym, fontsize, dpi) 
        return metrics

    def _get_info (self, font, sym, fontsize, dpi):
        'load the cmfont, metrics and glyph with caching'
        key = font, sym, fontsize, dpi
        tup = self.glyphd.get(key)

        if tup is not None: return tup

        basename = self.fontmap[font]

        if latex_to_bakoma.has_key(sym):
            basename, num = latex_to_bakoma[sym]
            num = self.charmaps[basename][num]
        elif len(sym) == 1:
            num = ord(sym)
        else:
            num = 0
            verbose.report_error('unrecognized symbol "%s"' % sym)

        #print sym, basename, num
        cmfont = self.fonts[basename]
        cmfont.set_size(fontsize, dpi)
        head  = cmfont.get_sfnt_table('head')
        glyph = cmfont.load_char(num)

        xmin, ymin, xmax, ymax = [val/64.0 for val in glyph.bbox]
        if basename == 'cmex10':
            offset =  glyph.height/64.0/2 + 256.0/64.0*dpi/72.0
            #offset = -(head['yMin']+512)/head['unitsPerEm']*10.
        else:
            offset = 0.
        metrics = Bunch(
            advance  = glyph.horiAdvance/64.0,
            height   = glyph.height/64.0,
            width    = glyph.width/64.0,
            xmin = xmin,
            xmax = xmax,
            ymin = ymin+offset,
            ymax = ymax+offset,
            )
        
        self.glyphd[key] = cmfont, metrics, glyph, offset
        return self.glyphd[key]

    def set_canvas_size(self, w, h):
        'Dimension the drawing canvas; may be a noop'
        self.width = int(w)
        self.height = int(h)
        for font in self.fonts.values():
            font.set_bitmap_size(int(w), int(h)) 

    def render(self, ox, oy, font, sym, fontsize, dpi):
        cmfont, metrics, glyph, offset = \
                self._get_info(font, sym, fontsize, dpi)

        if not self.usingSVG:
            cmfont.draw_glyph_to_bitmap(
                int(ox),  int(self.height - oy - metrics.ymax), glyph)
        else:
            oy += offset - 512/2048.*10.
            basename = self.fontmap[font]
            if latex_to_bakoma.has_key(sym):
                basename, num = latex_to_bakoma[sym]
                num = self.charmaps[basename][num]
            elif len(sym) == 1:
                num = ord(sym)
            else:
                num = 0
                print >>sys.stderr, 'unrecognized symbol "%s"' % sym
            self.svg_glyphs.append((basename, fontsize, num, ox, oy, metrics))
        

class BakomaPSFonts(Fonts):
    """
    Use the Bakoma postscript fonts for rendering to backend_ps
    """
    fnames = ('cmmi10', 'cmsy10', 'cmex10',
              'cmtt10', 'cmr10')
    # allocate a new set of fonts
    basepath = get_data_path()
    
    fontmap = { 'cal' : 'cmsy10',
                'rm'  : 'cmr10',
                'tt'  : 'cmtt10',
                'it'  : 'cmmi10',
                None  : 'cmmi10',
                }

    def __init__(self):
        self.glyphd = {}
        self.fonts = dict(
            [ (name, FT2Font(os.path.join(self.basepath, name) + '.ttf'))
              for name in self.fnames])

        self.charmaps = dict(
            [ (name, self.fonts[name].get_charmap()) for name in self.fnames])
        for font in self.fonts.values():
            font.clear()

    def _get_info (self, font, sym, fontsize, dpi):
        'load the cmfont, metrics and glyph with caching'
        key = font, sym, fontsize, dpi
        tup = self.glyphd.get(key)

        if tup is not None:
            return tup

        basename = self.fontmap[font]

        if latex_to_bakoma.has_key(sym):
            basename, num = latex_to_bakoma[sym]
            sym = self.fonts[basename].get_glyph_name(num)
            num = self.charmaps[basename][num]
        elif len(sym) == 1:
            num = ord(sym)
        else:
            num = 0
            sym = '.notdef'
            verbose.report_error('unrecognized symbol "%s, %d"' % (sym, num))

        if basename not in bakoma_fonts:
            bakoma_fonts.append(basename)
        cmfont = self.fonts[basename]
        cmfont.set_size(fontsize, dpi)
        head = cmfont.get_sfnt_table('head')
        glyph = cmfont.load_char(num)
        
        xmin, ymin, xmax, ymax = [val/64.0 for val in glyph.bbox]
        if basename == 'cmex10':
            offset = -(head['yMin']+512)/head['unitsPerEm']*10.
        else:
            offset = 0.
        metrics = Bunch(
            advance  = glyph.horiAdvance/64.0,
            height   = glyph.height/64.0,
            width    = glyph.width/64.0,
            xmin = xmin,
            xmax = xmax,
            ymin = ymin+offset,
            ymax = ymax+offset
            )

        self.glyphd[key] = basename, metrics, sym, offset
        return basename, metrics, '/'+sym, offset

    def set_canvas_size(self, w, h, pswriter):
        'Dimension the drawing canvas; may be a noop'
        self.width  = w
        self.height = h
        self.pswriter = pswriter
        
    def get_metrics(self, font, sym, fontsize, dpi):
        basename, metrics, sym, offset  = \
                self._get_info(font, sym, fontsize, dpi) 
        return metrics

    def render(self, ox, oy, font, sym, fontsize, dpi):
        fontname, metrics, glyphname, offset = \
                self._get_info(font, sym, fontsize, dpi)
        fontname = fontname.capitalize()
        if fontname == 'Cmex10':
            oy += offset - 512/2048.*10.
        
        ps = """/%(fontname)s findfont
%(fontsize)s scalefont
setfont
%(ox)f %(oy)f moveto
/%(glyphname)s glyphshow
""" % locals()
        self.pswriter.write(ps)


class Element:
    fontsize = 12
    dpi = 72
    font = 'it'
    _padx, _pady = 2, 2  # the x and y padding in points
    _scale = 1.0
    
    def __init__(self):
        # a dict mapping the keys above, below, subscript,
        # superscript, right to Elements in that position
        self.neighbors = {}
        self.ox, self.oy = 0, 0
        
    def advance(self):
        'get the horiz advance'
        raise NotImplementedError('derived must override')

    def height(self):
        'get the element height: ymax-ymin'
        raise NotImplementedError('derived must override')        

    def width(self):
        'get the element width: xmax-xmin'
        raise NotImplementedError('derived must override')        

    def xmin(self):
        'get the xmin of ink rect'
        raise NotImplementedError('derived must override')        

    def xmax(self):
        'get the xmax of ink rect'
        raise NotImplementedError('derived must override')        

    def ymin(self):
        'get the ymin of ink rect'
        raise NotImplementedError('derived must override')        

    def ymax(self):
        'get the ymax of ink rect'
        raise NotImplementedError('derived must override')        

    def set_font(self, font):
        'set the font (one of tt, it, rm , cal)'
        raise NotImplementedError('derived must override')        

    def render(self):
        'render to the fonts canvas'
        for element in self.neighbors.values():
            element.render()

    def set_origin(self, ox, oy):
        self.ox, self.oy = ox, oy

        # order matters! right needs to be evaled last
        keys = ('above', 'below', 'subscript', 'superscript', 'right')
        for loc in keys:
            element = self.neighbors.get(loc)
            if element is None: continue
            
            if loc=='above':
                nx = self.centerx() - element.width()/2.0
                ny = self.ymax() + self.pady() + (element.oy - element.ymax() + element.height())
                #print element, self.ymax(), element.height(), element.ymax(), element.ymin(), ny
            elif loc=='below':
                nx = self.centerx() - element.width()/2.0
                ny = self.ymin() - self.pady() - element.height()
            elif loc=='superscript':
                nx = self.xmax() 
                ny = self.ymax() - self.pady()
            elif loc=='subscript':
                nx = self.xmax() 
                ny = self.oy - 0.5*element.height() 
            elif loc=='right':
                nx = self.ox + self.advance()
                if self.neighbors.has_key('subscript'):
                    o = self.neighbors['subscript']
                    nx = max(nx, o.ox + o.advance())
                if self.neighbors.has_key('superscript'):
                    o = self.neighbors['superscript']
                    nx = max(nx, o.ox + o.advance())
                
                ny = self.oy
            element.set_origin(nx, ny)

    def set_size_info(self, fontsize, dpi):
        self.fontsize = self._scale*fontsize
        self.dpi = dpi
        for loc, element in self.neighbors.items():
            if loc in ('subscript', 'superscript'):
                element.set_size_info(0.7*self.fontsize, dpi)
            else:
                element.set_size_info(self.fontsize, dpi)

    def pady(self):
        return self.dpi/72.0*self._pady
        
    def padx(self):
        return self.dpi/72.0*self._padx

    def set_padx(self, pad):
        'set the y padding in points'
        self._padx = pad

    def set_pady(self, pad):
        'set the y padding in points'
        self._pady = pad

    def set_scale(self, scale):
        'scale the element by scale'
        self._scale = scale
        
    def centerx(self):
        return 0.5 * (self.xmax() + self.xmin() ) 

    def centery(self):
        return 0.5 * (self.ymax() + self.ymin() ) 

    def __repr__(self):
        return str(self.__class__) + str(self.neighbors)
               
class SpaceElement(Element):
    'blank horizontal space'
    def __init__(self, space, height=0):
        """
        space is the amount of blank space in fraction of fontsize
        height is the height of the space in fraction of fontsize
        """
        Element.__init__(self)
        self.space = space
        self._height = height

    def advance(self):
        'get the horiz advance'
        return self.dpi/72.0*self.space*self.fontsize

    def height(self):
        'get the element height: ymax-ymin'
        return self._height*self.dpi/72.0*self.fontsize

    def width(self):
        'get the element width: xmax-xmin'
        return self.advance()
        
    def xmin(self):
        'get the minimum ink in x'
        return self.ox 

    def xmax(self):
        'get the max ink in x'
        return self.ox + self.advance()

    def ymin(self):
        'get the minimum ink in y'
        return self.oy 

    def ymax(self):
        'get the max ink in y'
        return self.oy + self.height()

    def set_font(self, f):
        # space doesn't care about font, only size
        pass
    
class SymbolElement(Element):
    def __init__(self, sym):
        Element.__init__(self)
        self.sym = sym

    def set_font(self, font):
        'set the font (one of tt, it, rm , cal)'
        self.font = font

    def set_origin(self, ox, oy):
        Element.set_origin(self, ox, oy)

    def set_size_info(self, fontsize, dpi):
        Element.set_size_info(self, fontsize, dpi)
        self.metrics = Element.fonts.get_metrics(
            self.font, self.sym, self.fontsize, dpi)

    def advance(self):
        'get the horiz advance'
        return self.metrics.advance


    def height(self):
        'get the element height: ymax-ymin'
        return self.metrics.height

    def width(self):
        'get the element width: xmax-xmin'
        return self.metrics.width
        
    def xmin(self):
        'get the minimum ink in x'
        return self.ox + self.metrics.xmin

    def xmax(self):
        'get the max ink in x'
        return self.ox + self.metrics.xmax

    def ymin(self):
        'get the minimum ink in y'
        return self.oy + self.metrics.ymin

    def ymax(self):
        'get the max ink in y'
        return self.oy + self.metrics.ymax

    def render(self):
        'render to the fonts canvas'
        Element.render(self)
        Element.fonts.render(
            self.ox, self.oy, 
            self.font, self.sym, self.fontsize, self.dpi)

    def __repr__(self):
        return self.sym

    
class GroupElement(Element):
    """
    A group is a collection of elements
    """
    def __init__(self, elements):
        Element.__init__(self)
        self.elements = elements
        for i in range(len(elements)-1):
            self.elements[i].neighbors['right'] = self.elements[i+1]

    def set_font(self, font):
        'set the font (one of tt, it, rm , cal)'
        for element in self.elements:
            element.set_font(font)
        
    def set_size_info(self, fontsize, dpi):        
        self.elements[0].set_size_info(self._scale*fontsize, dpi)
        Element.set_size_info(self, fontsize, dpi)

    def set_origin(self, ox, oy):
        self.elements[0].set_origin(ox, oy)
        Element.set_origin(self, ox, oy)


    def advance(self):
        'get the horiz advance'
        return self.elements[-1].xmax() - self.elements[0].ox 


    def height(self):
        'get the element height: ymax-ymin'
        ymax = max([e.ymax() for e in self.elements])
        ymin = min([e.ymin() for e in self.elements])
        return ymax-ymin
    
    def width(self):
        'get the element width: xmax-xmin'
        xmax = max([e.xmax() for e in self.elements])
        xmin = min([e.xmin() for e in self.elements])
        return xmax-xmin

    def render(self):
        'render to the fonts canvas'
        Element.render(self)
        self.elements[0].render()

    def xmin(self):
        'get the minimum ink in x'
        return min([e.xmin() for e in self.elements])

    def xmax(self):
        'get the max ink in x'
        return max([e.xmax() for e in self.elements])        

    def ymin(self):
        'get the minimum ink in y'
        return max([e.ymin() for e in self.elements])        

    def ymax(self):
        'get the max ink in y'
        return max([e.ymax() for e in self.elements])                

    def __repr__(self):
        return 'Group: [ %s ]' % ' '.join([str(e) for e in self.elements])

class ExpressionElement(GroupElement):
    """
    The entire mathtext expression
    """

    def __repr__(self):
        return 'Expression: [ %s ]' % ' '.join([str(e) for e in self.elements])


class Handler:
    symbols = []
    
    def clear(self):
        self.symbols = []

    def expression(self, s, loc, toks):
        self.expr = ExpressionElement(toks)
        return loc, [self.expr]

    def space(self, s, loc, toks):
        assert(len(toks)==1)

        if toks[0]==r'\ ': num = 0.30 # 30% of fontsize        
        elif toks[0]==r'\/': num = 0.1 # 10% of fontsize
        else:  # vspace
            num = float(toks[0][1]) # get the num out of \hspace{num}
            
        element = SpaceElement(num)
        self.symbols.append(element)
        return loc, [element]

    def symbol(self, s, loc, toks):

        assert(len(toks)==1)

        s  = toks[0]
        #~ print 'sym', toks[0]
        if charOverChars.has_key(s):
            under, over, pad = charOverChars[s]
            font, tok, scale = under
            sym = SymbolElement(tok)
            if font is not None:
                sym.set_font(font)
            sym.set_scale(scale)
            sym.set_pady(pad)

            font, tok, scale = over
            sym2 = SymbolElement(tok)
            if font is not None:
                sym2.set_font(font)
            sym2.set_scale(scale)

            sym.neighbors['above'] = sym2
            self.symbols.append(sym2)
        else:
            sym = SymbolElement(toks[0])
        self.symbols.append(sym)
        
        return loc, [sym]

    def composite(self, s, loc, toks):

        assert(len(toks)==1)
        where, sym0, sym1 = toks[0]
        #keys = ('above', 'below', 'subscript', 'superscript', 'right')
        if where==r'\over':
            sym0.neighbors['above'] = sym1
        elif where==r'\under':
            sym0.neighbors['below'] = sym1
            
        self.symbols.append(sym0)
        self.symbols.append(sym1)        
        
        return loc, [sym0]    

    def accent(self, s, loc, toks):

        assert(len(toks)==1)
        accent, sym = toks[0]

        d = {
            r'\hat'   : r'\circumflexaccent',
            r'\breve' : r'\combiningbreve',
            r'\bar'   : r'\combiningoverline',
            r'\grave' : r'\combininggraveaccent',
            r'\acute' : r'\combiningacuteaccent',
            r'\ddot'  : r'\combiningdiaeresis',
            r'\tilde' : r'\combiningtilde',
            r'\dot'   : r'\combiningdotabove',            
            r'\vec'   : r'\combiningrightarrowabove',                        
            r'\"'     : r'\combiningdiaeresis',
            r"\`"     : r'\combininggraveaccent',
            r"\'"     : r'\combiningacuteaccent',
            r'\~'     : r'\combiningtilde',
            r'\.'     : r'\combiningdotabove',
            r'\^'   : r'\circumflexaccent',            
             }
        above = SymbolElement(d[accent])
        sym.neighbors['above'] = above
        sym.set_pady(1)
        self.symbols.append(above)
        return loc, [sym]    

    def group(self, s, loc, toks):
        assert(len(toks)==1)
        #print 'grp', toks
        grp = GroupElement(toks[0])
        return loc, [grp]

    def font(self, s, loc, toks):

        assert(len(toks)==1)
        name, grp = toks[0]
        #print 'fontgrp', toks
        grp.set_font(name[1:])  # suppress the slash
        return loc, [grp]

    def subscript(self, s, loc, toks):
        assert(len(toks)==1)
        #print 'subsup', toks
        if len(toks[0])==2:
            under, next = toks[0]
            prev = SpaceElement(0)
        else:
            prev, under, next = toks[0]            

        if self.is_overunder(prev):
            prev.neighbors['below'] = next
        else:
            prev.neighbors['subscript'] = next

        return loc, [prev]

    def is_overunder(self, prev):
        return isinstance(prev, SymbolElement) and overunder.has_key(prev.sym)
    
    def superscript(self, s, loc, toks):
        assert(len(toks)==1)
        #print 'subsup', toks
        if len(toks[0])==2:
            under, next = toks[0]
            prev = SpaceElement(0,0.6)
        else:
            prev, under, next = toks[0]
        if self.is_overunder(prev):
            prev.neighbors['above'] = next
        else:
            prev.neighbors['superscript'] = next

        return loc, [prev]

    def subsuperscript(self, s, loc, toks):
        assert(len(toks)==1)
        #print 'subsup', toks
        prev, undersym, down, oversym, up = toks[0]

        if self.is_overunder(prev):
            prev.neighbors['below'] = down
            prev.neighbors['above'] = up
        else:
            prev.neighbors['subscript'] = down
            prev.neighbors['superscript'] = up

        return loc, [prev]



handler = Handler()

lbrace = Literal('{').suppress()
rbrace = Literal('}').suppress()
lbrack = Literal('[')
rbrack = Literal(']')
lparen = Literal('(')
rparen = Literal(')')
grouping = lbrack | rbrack | lparen | rparen

bslash = Literal('\\')


langle = Literal('<')
rangle = Literal('>')
equals = Literal('=')
relation = langle | rangle | equals

colon =  Literal(':')
comma =  Literal(',')
period =  Literal('.')
semicolon =  Literal(';')
exclamation =  Literal('!')

punctuation = colon | comma | period | semicolon 

at =  Literal('@')
percent =  Literal('%')
ampersand =  Literal('&')
misc = exclamation | at | percent | ampersand

over = Literal('over')
under = Literal('under')
#~ composite = over | under
overUnder = over | under

accent = Literal('hat') | Literal('check') | Literal('dot') | \
         Literal('breve') | Literal('acute') | Literal('ddot') | \
         Literal('grave') | Literal('tilde') | Literal('bar') | \
         Literal('vec') | Literal('"') | Literal("`") | Literal("'") |\
         Literal('~') | Literal('.') | Literal('^')
         



number = Combine(Word(nums) + Optional(Literal('.')) + Optional( Word(nums) ))

plus = Literal('+')
minus = Literal('-')
times = Literal('*')
div = Literal('/')
binop = plus | minus | times | div


roman      = Literal('rm')
cal        = Literal('cal')
italics    = Literal('it')
typewriter = Literal('tt')
fontname   = roman | cal | italics | typewriter

texsym = Combine(bslash + Word(alphanums) + NotAny(lbrace))

char = Word(alphanums + ' ', exact=1).leaveWhitespace()

space = (Literal(r'\ ') | Literal(r'\/') | Group(Literal(r'\hspace{') + number + Literal('}'))).setParseAction(handler.space).setName('space')

#~ symbol = (texsym ^ char ^ binop ^ relation ^ punctuation ^ misc ^ grouping  ).setParseAction(handler.symbol).leaveWhitespace()
symbol = (texsym | char | binop | relation | punctuation | misc | grouping  ).setParseAction(handler.symbol).leaveWhitespace()

subscript = Forward().setParseAction(handler.subscript).setName("subscript")
superscript = Forward().setParseAction(handler.superscript).setName("superscript")
subsuperscript = Forward().setParseAction(handler.subsuperscript).setName("subsuperscript")

font = Forward().setParseAction(handler.font).setName("font")


group = Group( lbrace + OneOrMore(symbol^subscript^superscript^subsuperscript^space^font) + rbrace).setParseAction(handler.group).setName("group")
#~ group = Group( lbrace + OneOrMore(subsuperscript | subscript | superscript | symbol | space ) + rbrace).setParseAction(handler.group).setName("group")

#composite = Group( Combine(bslash + composite) + lbrace + symbol + rbrace + lbrace + symbol + rbrace).setParseAction(handler.composite).setName("composite")
#~ composite = Group( Combine(bslash + composite) + group + group).setParseAction(handler.composite).setName("composite")
composite = Group( Combine(bslash + overUnder) + group + group).setParseAction(handler.composite).setName("composite")



accent = Group( Combine(bslash + accent) + Optional(lbrace) + symbol + Optional(rbrace)).setParseAction(handler.accent).setName("accent")


symgroup = font ^ group ^ symbol 

subscript << Group( Optional(symgroup) + Literal('_') + symgroup  )
superscript << Group( Optional(symgroup) + Literal('^') + symgroup  )
subsuperscript << Group( symgroup + Literal('_') + symgroup + Literal('^') + symgroup  )

font << Group( Combine(bslash + fontname) + group)



expression = OneOrMore(
    space ^ font ^ accent ^ symbol ^ subscript ^ superscript ^ subsuperscript ^ group ^ composite  ).setParseAction(handler.expression).setName("expression")
#~ expression = OneOrMore(
    #~ group | composite | space | font | subsuperscript | subscript | superscript | symbol ).setParseAction(handler.expression).setName("expression")

####




def math_parse_s_ft2font(s, dpi, fontsize, angle=0):
    """
    Parse the math expression s, return the (bbox, fonts) tuple needed
    to render it.

    fontsize must be in points

    return is width, height, fonts
    """

    major, minor1, minor2, tmp, tmp = sys.version_info
    if major==2 and minor1==2:
        raise SystemExit('mathtext broken on python2.2.  We hope to get this fixed soon')

    cacheKey = (s, dpi, fontsize, angle)
    s = s[1:-1]  # strip the $ from front and back
    if math_parse_s_ft2font.cache.has_key(cacheKey):
        w, h, bfonts = math_parse_s_ft2font.cache[cacheKey]
        return w, h, bfonts.fonts.values()

    bakomaFonts = BakomaTrueTypeFonts()
    Element.fonts = bakomaFonts
    handler.clear()
    expression.parseString( s )

    handler.expr.set_size_info(fontsize, dpi)

    # set the origin once to allow w, h compution
    handler.expr.set_origin(0, 0)
    xmin = min([e.xmin() for e in handler.symbols])
    xmax = max([e.xmax() for e in handler.symbols])
    ymin = min([e.ymin() for e in handler.symbols])
    ymax = max([e.ymax() for e in handler.symbols])

    # now set the true origin - doesn't affect with and height
    w, h =  xmax-xmin, ymax-ymin
    # a small pad for the canvas size
    w += 2
    h += 2

    handler.expr.set_origin(0, h-ymax)

    Element.fonts.set_canvas_size(w,h)
    handler.expr.render()
    handler.clear()

    math_parse_s_ft2font.cache[cacheKey] = w, h, bakomaFonts
    return w, h, bakomaFonts.fonts.values()

math_parse_s_ft2font.cache = {}

def math_parse_s_ft2font_svg(s, dpi, fontsize, angle=0):
    """
    Parse the math expression s, return the (bbox, fonts) tuple needed
    to render it.

    fontsize must be in points

    return is width, height, fonts
    """

    major, minor1, minor2, tmp, tmp = sys.version_info
    if major==2 and minor1==2:
        print >> sys.stderr, 'mathtext broken on python2.2.  We hope to get this fixed soon'
        sys.exit()
    cacheKey = (s, dpi, fontsize, angle)
    s = s[1:-1]  # strip the $ from front and back
    if math_parse_s_ft2font_svg.cache.has_key(cacheKey):
        w, h, svg_glyphs = math_parse_s_ft2font_svg.cache[cacheKey]
        return w, h, svg_glyphs

    bakomaFonts = BakomaTrueTypeFonts(useSVG=True)
    Element.fonts = bakomaFonts
    handler.clear()
    expression.parseString( s )

    handler.expr.set_size_info(fontsize, dpi)

    # set the origin once to allow w, h compution
    handler.expr.set_origin(0, 0)
    xmin = min([e.xmin() for e in handler.symbols])
    xmax = max([e.xmax() for e in handler.symbols])
    ymin = min([e.ymin() for e in handler.symbols])
    ymax = max([e.ymax() for e in handler.symbols])

    # now set the true origin - doesn't affect with and height
    w, h =  xmax-xmin, ymax-ymin
    # a small pad for the canvas size
    w += 2
    h += 2

    handler.expr.set_origin(0, h-ymax)

    Element.fonts.set_canvas_size(w,h)
    handler.expr.render()
    handler.clear()

    math_parse_s_ft2font_svg.cache[cacheKey] = w, h, bakomaFonts.svg_glyphs
    return w, h, bakomaFonts.svg_glyphs

math_parse_s_ft2font_svg.cache = {}


def math_parse_s_ps(s, dpi, fontsize):
    """
    Parse the math expression s, return the (bbox, fonts) tuple needed
    to render it.

    fontsize must be in points

    return is width, height, fonts

    """
    cacheKey = (s, dpi, fontsize)
    s = s[1:-1]  # strip the $ from front and back
    if math_parse_s_ps.cache.has_key(cacheKey):
        w, h, pswriter = math_parse_s_ps.cache[cacheKey]
        return w, h, pswriter

    bakomaFonts = BakomaPSFonts()
    Element.fonts = bakomaFonts
    handler.clear()
    expression.parseString( s )

    handler.expr.set_size_info(fontsize, dpi)

    # set the origin once to allow w, h compution
    handler.expr.set_origin(0, 0)
    xmin = min([e.xmin() for e in handler.symbols])
    xmax = max([e.xmax() for e in handler.symbols])
    ymin = min([e.ymin() for e in handler.symbols])
    ymax = max([e.ymax() for e in handler.symbols])

    # now set the true origin - doesn't affect with and height
    w, h =  xmax-xmin, ymax-ymin
    # a small pad for the canvas size
    w += 2
    h += 2

    handler.expr.set_origin(0, h-ymax)

    pswriter = StringIO()
    Element.fonts.set_canvas_size(w, h, pswriter)
    handler.expr.render()
    handler.clear()

    math_parse_s_ps.cache[cacheKey] = w, h, pswriter
    return w, h, pswriter

math_parse_s_ps.cache = {}

if __name__=='___main__':
    
    stests = [ 
            r'$dz/dt \/ = \/ \gamma x^2 \/ + \/ \rm{sin}(2\pi y+\phi)$',
            r'$dz/dt \/ = \/ \gamma xy^2 \/ + \/ \rm{s}(2\pi y+\phi)$',
            r'$x^1 2$',
            r'$\alpha_{i+1}^j \/ = \/ \rm{sin}(2\pi f_j t_i) e^{-5 t_i/\tau}$',
            r'$\cal{R}\prod_{i=\alpha_{i+1}}^\infty a_i\rm{sin}(2 \pi f x_i)$',
            r'$\bigodot \bigoplus \cal{R} a_i\rm{sin}(2 \pi f x_i)$',
            r'$x_i$',
            r'$5\/\angstrom\hspace{ 2.0 }\pi$',
            r'$x+1$',
            r'$i$',
            r'$i^j$',
            ]
    #~ w, h, fonts = math_parse_s_ft2font(s, 20, 72)
    for s in stests:
        try:
            print s
            print (expression + StringEnd()).parseString( s[1:-1] )
        except ParseException, pe:
            print "*** ERROR ***", pe.msg
            print s
            print 'X' + (' '*pe.loc)+'^'
            # how far did we get?
            print expression.parseString( s[1:-1] )
        print

    #w, h, fonts = math_parse_s_ps(s, 20, 72)


if __name__=='__main__':
    s = 'i'

    Element.fonts = DummyFonts()
    handler.clear()
    expression.parseString( s )

    handler.expr.set_size_info(12, 72)

    # set the origin once to allow w, h compution
    handler.expr.set_origin(0, 0)
    for e in handler.symbols:
        assert(hasattr(e, 'metrics'))
