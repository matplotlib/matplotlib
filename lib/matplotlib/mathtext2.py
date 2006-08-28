r"""
Supported commands:
-------------------
 * _, ^, to any depth
 * commands for typesetting functions (\sin, \cos etc.),
 * commands for changing the current font (\rm, \cal etc.),
 * Space/kern commands "\ ", \thinspace
 * \frac

Small TO-DO's:
--------------
 * Display braces etc. \} not working (displaying wierd characters) etc.
 * better placing of sub/superscripts. F_1^1y_{1_{2_{3_{4}^3}}3}1_23
 * implement crampedness (or is it smth. else?). y_1 vs. y_1^1
 * add better italic correction. F^1
 * implement other space/kern commands

TO-DO's:
--------
 * \over, \above, \choose etc.
 * Add support for other backends

"""
import os
from math import fabs

from matplotlib import get_data_path, rcParams
from matplotlib._mathtext_data import tex2uni
from matplotlib.ft2font import FT2Font, KERNING_DEFAULT
from matplotlib.font_manager import findSystemFonts
from copy import deepcopy

_path = get_data_path()
faces = ('mit', 'rm', 'tt', 'cal', 'nonascii')

filenamesd = {}
fonts = {}

# Filling the above dicts
for face in faces:
    # The filename without the path
    barefname = rcParams['mathtext.' + face]
    base, ext = os.path.splitext(barefname)
    if not ext:
        ext = '.ttf'
        barefname = base + ext
    # First, we search for the font in the system font dir
    for fname in findSystemFonts(fontext=ext[1:]):
        if fname.endswith(barefname):
            filenamesd[face] = fname
            break
    # We check if the for loop above had success. If it failed, we try to
    # find the font in the mpl-data dir
    if not face in filenamesd:
        filenamesd[face] = os.path.join(_path, barefname)
    fonts[face] = FT2Font(filenamesd[face])

esc_char = '\\'
# Grouping delimiters
begin_group_char = '{'
end_group_char = '}'
dec_delim = '.'
word_delim = ' '
mathstyles = ["display", "text", "script", "scriptscript"]
modes = ["mathmode", "displaymathmode"]

# Commands
scripts = ("_", "^")
functions = ("sin", "tan", "cos", "exp", "arctan", "arccos", "arcsin", "cot")

# Commands that change the environment (in the current scope)
setters = faces
# Maximum number of nestings (groups within groups)
max_depth = 10

#~ environment = {
#~ "mode": "mathmode",
#~ "mathstyle" : "display",
#~ "cramped" : False,
#~ # We start with zero scriptdepth (should be incremented by a Scripted
#~ # instance)
#~ "scriptdepth" : 0, 
#~ "face" : None,
#~ "fontsize" : 12,
#~ "dpi" : 100,
#~ }


# _textclass can be unicode or str.
_textclass = unicode


# Exception classes
class TexParseError(Exception):
    pass

# Helper classes
class Scriptfactors(dict):
    """Used for returning the factor with wich you should multiply the
    fontsize to get the font size of the script
    
    """
    _scriptfactors = {
                        0 : 1,  # Normal text
                        1: 0.8, # Script
                        2: 0.6, # Scriptscript
                        # For keys > 3 returns 0.6
                    }

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise KeyError("Integer value needed for scriptdepth")
        if key < 0:
            raise KeyError("scriptdepth must be positive")
        if key in self._scriptfactors:
            return self._scriptfactors[key]
        else:
            # Maximum depth of scripts is 2 (scriptscript)
            return self._scriptfactors[2]

scriptfactors = Scriptfactors()


class Environment:
    """Class used for representing the TeX environment variables"""
    def __init__(self):
        self.mode = "mathmode"
        self.mathstyle = "display"
        self.cramped = False
        # We start with zero scriptdepth (should be incremented by a Scripted
        # instance)
        self.scriptdepth = 0
        self.face = None
        self.fontsize = 12
        self.dpi = 100

    def copy(self):
        return deepcopy(self)

# The topmost environment
environment = Environment()


# Helper functions used by the parser
def parse_tex(texstring):
    texstring = normalize_tex(texstring)
    _parsed = to_list(texstring)
    #_parsed = Hbox(_parsed)
    return _parsed

def remove_comments(texstring):
    # TO-DO
    return texstring

def group_split(texstring):
    """Splits the string into three parts based on the grouping delimiters,
    and returns them as a list.
    """
    if texstring == begin_group_char + end_group_char:
        return '', [], ''
    length = len(texstring)
    i = texstring.find(begin_group_char)
    if i == -1:
        return texstring, '', ''
    pos_begin = i
    count = 1
    num_groups = 0
    while count != 0:
        i = i + 1
        # First we check some things
        if num_groups > max_depth:
            message = "Maximum number of nestings reached. Too many groups"
            raise TexParseError(message)
        if i == length:
            message = "Group not closed properly"
            raise TexParseError(message)

        if texstring[i] == end_group_char:
            count -= 1
        elif texstring[i] == begin_group_char:
            num_groups += 1
            count += 1
    before = texstring[:pos_begin]
    if pos_begin + 1 == i:
        grouping = []
    else:
        grouping = texstring[pos_begin + 1:i]
    after = texstring[i + 1:]
    return before, grouping, after

def break_up_commands(texstring):
    """Breaks up a string (mustn't contain any groupings) into a list
    of commands and pure text.
    """
    result = []
    if not texstring:
        return result
    _texstrings = texstring.split(esc_char)
    for i, _texstring in enumerate(_texstrings):
        _command, _puretext = split_command(_texstring)
        if i == 0 and _texstrings[0]:
            # Case when the first command is a not a command but text
            result.extend([c for c in _command])
            result.extend(_puretext)
            continue
        if _command:
            result.append(esc_char + _command)
        if _puretext:
            if _puretext[0] == word_delim:
                _puretext = _puretext[1:]
            result.extend(_puretext)
    return result

def split_command(texstring):
    """Splits a texstring into a command part and a pure text (as a list) part
    
    """
    if not texstring:
        return "", []
    _puretext = []
    _command, _rest = get_first_word(texstring)
    if not _command:
        _command = texstring[0]
        _rest = texstring[1:]
    _puretext = [c for c in _rest]
    #~ while True:
        #~ _word, _rest = get_first_word(_rest)
        #~ if _word:
            #~ _puretext.append(_word)
        #~ if _rest:
            #~ _puretext.extend(_rest[0])
            #~ if len(_rest) == 1:
                #~ break
            #~ _rest = _rest[1:]
        #~ else:
            #~ break
    return _command, _puretext

def get_first_word(texstring):
    _word = ""
    i = 0
    _length = len(texstring)
    if _length == 0:
        return "", ""
    if texstring[0].isalpha():
        while _length > i and texstring[i].isalpha():
            _word += texstring[i]
            i = i + 1
    elif texstring[0].isdigit():
        while _length > i and (texstring[i].isdigit()):
            _word += texstring[i]
            i = i + 1
        
    return _word, texstring[i:]

def to_list(texstring):
    """Parses the normalized tex string and returns a list. Used recursively.
    """
    result = []
    if not texstring:
        return result
    # Checking for groupings: begin_group_char...end_group_char
    before, grouping, after = group_split(texstring)
    #print "Before: ", before, '\n', grouping, '\n', after

    if before:
        result.extend(break_up_commands(before))
    if grouping or grouping == []:
        result.append(to_list(grouping))
    if after:
        result.extend(to_list(after))

    return result

def normalize_tex(texstring):
    """Normalizes the whole TeX expression (that is: prepares it for
    parsing)"""
    texstring = remove_comments(texstring)
    # Removing the escaped escape character (replacing it)
    texstring = texstring.replace(esc_char + esc_char, esc_char + 'backslash')
    
    # Removing the escaped scope/grouping characters
    texstring = texstring.replace(esc_char + begin_group_char, esc_char + 'lbrace')
    texstring = texstring.replace(esc_char + end_group_char, esc_char + 'rbrace')

    # Now we should have a clean expression, so we check if all the grouping
    # are OK (every begin_group_char should have a matching end_group_char)
    # TO-DO

    # Replacing all space-like characters with a single space word_delim
    texstring = word_delim.join(texstring.split())

    # Removing unnecessary white space
    texstring = word_delim.join(texstring.split())
    return texstring

def is_command(item):
    try:
        return item.startswith(esc_char)
    except AttributeError:
        return False


# Helper functions used by the renderer
def get_frac_bar_height(env):
    # TO-DO: Find a better way to calculate the height of the rule
    c = TexCharClass(env, ".")
    return c.height

def get_font(env, item):
    env = env.copy()
    # TO-DO: Perhaps this should be done somewhere else
    if not env.face:
        env.face = "rm"
    fontsize = env.fontsize * scriptfactors[env.scriptdepth]
    dpi = env.dpi
    font = fonts[env.face]

    font.set_size(fontsize, dpi)
    return font
    #~ font = FT2Font(filenamesd[face])
    #~ if fonts:
        #~ fonts[max(fonts.keys()) + 1] = font
    #~ else:
        #~ fonts[1] = font
def infer_face(env, item):
    if item.isalpha():
        if env.mode == "mathmode" and item < "z":
            face = "mit"
        else:
            # TO-DO: Perhaps change to 'rm'
            face = "nonascii"
    elif item.isdigit():
        face = "rm"
    elif ord(item) < 256:
        face = "rm"
    else:
        face = "nonascii"
    return face

def get_space(env):
    _env = env.copy()
    if not _env.face:
        _env.face = "rm"
    space = TexCharClass(_env, " ")
    return space

def get_kern(env, first, second):
    try:
        if first.env == second.env and\
            isinstance(first,TexCharClass) and isinstance(first, TexCharClass):
            font = get_font(first.env, first)
            advance = -font.get_kerning(first.uniindex, second.uniindex,
                                        KERNING_DEFAULT)/64.0
            return Kern(env, advance)
        else:
            return Kern(env, 0)
    except AttributeError:
        return Kern(env, 0)


# Classes used for renderering
class Renderer:
    """Abstract class that implements the rendering methods"""
    def __init__(self, env):
        # We initialize all the values to 0.0
        self.xmin, self.ymin, self.xmax, self.ymax = (0.0,)*4
        (self.width, self.height, self.advance, self.bearingx,
            self.bearingy)= (0.0,)*5
        self.env = env

    def __render__(self):
        raise NotImplementedError("Derived must override")

class Hbox(Renderer):
    """A class that corresponds to a TeX hbox."""
    def __init__(self, env, texlist=[]):
        Renderer.__init__(self, env)
        self.items = texlist
        if not self.items:
            # empty group
            return
        for item in self.items:
            # The xmin remains the xmin of the first item
            self.xmax += item.advance

            self.bearingy = max((item.bearingy, self.bearingy))
            self.ymax = max((item.ymax, self.ymax))
            self.ymin = min((item.ymin, self.ymin))
            self.advance += item.advance
            previous = item
        # The bearingx of the whole group is the bearingx of the first item.
        first = self.items[0]
        self.bearingx = first.bearingx
        self.xmin = first.xmin

        last = self.items[-1]
        self.xmax = self.advance + fabs(last.advance - last.xmax)
        self.width = self.xmax - self.xmin
        self.height = self.ymax - self.ymin

    def render(self, x, y):
        for item in self.items:
            item.render(x, y)
            x += item.advance


class Scripted(Renderer):
    """Used for creating elements that have sub/superscripts"""
    def __init__(self, env, nuc=None, type="ord", sub=None,
                sup=None):
        Renderer.__init__(self, env)
        if not nuc:
            nuc = Hbox([])
        if not sub:
            sub = Hbox([])
        if not sup:
            sup = Hbox([])
        self.nuc = nuc
        self.sub = sub
        self.sup = sup
        self.type = type
        # Heuristics for figuring out how much the subscripts origin has to be
        # below the origin of the nucleus (the descent of the letter "j").
        # TO-DO: Change with a better alternative. Not working: F_1^1y_1
        c = TexCharClass(env, "j")
        C = TexCharClass(env, "M")

        self.subpad = c.height - c.bearingy
        # If subscript is complex (i.e. a large Hbox - fraction etc.)
        # we have to aditionaly lower the subscript
        if sub.ymax > (C.height/2.1 + self.subpad):
            self.subpad = sub.ymax - C.height/2.1
            
        #self.subpad = max(self.subpad)
        #self.subpad = 0.5*sub.height
        # Similar for the superscript
        self.suppad = max(nuc.height/1.9, C.ymax/1.9) - sup.ymin# - C.bearingy


        #self.advance = nuc.advance + max((sub.advance, sup.advance))

        self.xmin = nuc.xmin

        self.xmax = max(nuc.advance, nuc.bearingx + nuc.width) +\
        max((sub.advance, sub.bearingx + sub.width,
            sup.advance, sup.bearingx + sup.width))# - corr

        self.ymin = min(nuc.ymin, -self.subpad + sub.ymin)

        self.ymax = max((nuc.ymax, self.suppad + sup.bearingy))

        # The bearing of the whole element is the bearing of the nucleus
        self.bearingx = nuc.bearingx
        self.advance = self.xmax
        # Heruistics. Feel free to change
        self.bearingy = self.ymax

        self.width = self.xmax - self.xmin
        self.height = self.ymax - self.ymin

    def render(self, x, y):
        nuc, sub, sup = self.nuc, self.sub, self.sup
        nx = x
        ny = y

        subx = x + max(nuc.advance, nuc.bearingx + nuc.width)# + sub.bearingx
        suby = y + self.subpad# - subfactor*self.env.fontsize

        supx = x + max(nuc.advance, nuc.bearingx + nuc.width)# + sup.bearingx
        supy = y - self.suppad# + 10#subfactor*self.env.fontsize

        self.nuc.render(nx, ny)
        self.sub.render(subx, suby)
        self.sup.render(supx, supy)

    def __repr__(self):
        tmp = (repr(i) for i in [self.env, self.nuc, self.type,
            self.sub, self.sup])
        tmp = tuple(tmp)
        return "Scripted(env=%s,nuc=%s, type=%s, \
sub=%s, sup=%s)"%tmp


class Fraction(Renderer):
    """A class for rendering a fraction."""

    def __init__(self, env, num, den):
        Renderer.__init__(self, env)
        self.numer = num
        self.denom = den

        # TO-DO: Find a better way to implement the fraction bar
        self.pad = get_frac_bar_height(self.env)
        pad = self.pad
        self.bar = Line(env.copy(), max(num.width, den.width) + 2*pad, pad)
        #~ self.bar.bearingx = pad
        #~ self.bar.advance = self.bar.width + 2*pad
        #~ self.bar.bearingy = pad + pad

        self.xmin = 0
        #self.xmax = self.bar.advance
        self.xmax = self.bar.width# + 2*pad

        self.ymin = -(2*pad + den.height)
        self.ymax = 2*pad + num.height
        # The amount by which we raise the bar (the whole fraction)
        # of the bottom (origin)
        # TO-DO: Find a better way to implement it
        _env = env.copy()
        _env.face = "rm"
        c = TexCharClass(_env, "1")
        self.barpad = 1./3.*c.height
        self.ymin += self.barpad
        self.ymax += self.barpad
        
        self.width = self.xmax - self.xmin
        self.height = self.ymax - self.ymin
        #print self.width, self.height
        
        #self.bearingx = pad
        self.bearingx = 0
        self.bearingy = self.ymax
        #self.advance = self.bar.advance
        self.advance = self.xmax

    def render(self, x, y):
        y -= self.barpad
        pad = self.pad
        #print self.bar.xmax, self.bar.xmin, self.bar.ymin, self.bar.ymax
        self.bar.render(x, y)

        nx = x - self.numer.bearingx + (self.width - self.numer.width)/2.
        ny = y - 2*pad - (self.numer.height - self.numer.ymax)
        self.numer.render(nx, ny)
        
        dx = x - self.denom.bearingx+ (self.width - self.denom.width)/2.
        dy = y + 2*pad + self.denom.bearingy
        self.denom.render(dx, dy)


# Primitives
class TexCharClass(Renderer):
    """A class that implements rendering of a single character."""
    def __init__(self, env, char, uniindex=None):
        #Renderer.__init__(self, env)
        self.env = env
        # uniindex is used to override ord(char) (needed on platforms where
        # there is only BMP support for unicode, i.e. windows)
        msg = "A char (string with length == 1) is needed"
        if isinstance(_textclass(char), _textclass) and len(char) == 1:
            self.char = char
        else:
            raise ValueError(msg)
        if not uniindex:
            self.uniindex = ord(char)
        else:
            if isinstance(uniindex, int):
                self.uniindex = uniindex
            else:
                raise ValueError("uniindex must be an int")
        font = get_font(self.env, self)
        glyph = font.load_char(self.uniindex)
        self.glyph = glyph
        self.xmin, self.ymin, self.xmax, self.ymax = [
            val/64.0 for val in self.glyph.bbox]

        self.width = glyph.width/64.0
        self.height = glyph.height/64.0
        self.advance = glyph.horiAdvance/64.0
        self.bearingx = glyph.horiBearingX/64.0
        self.bearingy = glyph.horiBearingY/64.0

    def render(self, x, y):
        x += self.bearingx
        #y -= self.bearingy
        y -= self.ymax
        #y -= (self.height - self.bearingy)
        #print x, y
        font = get_font(self.env, self)
        font.draw_glyph_to_bitmap(x, y, self.glyph)


class Kern(Renderer):
    """Class that implements the rendering of a Kern."""

    def __init__(self, env, advance):
        Renderer.__init__(self, env)
        self.width = advance
        self.advance = advance

    def render(self, x, y):
        pass

    def __repr__(self):
        return "Kern(%s, %s)"%(self.env, self.advance)

class Line(Renderer):
    """Class that implements the rendering of a line."""

    def __init__(self, env, width, height):
        Renderer.__init__(self, env)
        self.ymin = -height/2.
        self.xmax = width
        self.ymax = height/2.

        self.width = width
        self.height = height
        self.advance = width
        self.bearingy = self.ymax

    def render(self, x, y):
        font = get_font(self.env, self)
        coords = (x + self.xmin, y + self.ymin, x + self.xmax,
                                                        y + self.ymax)
        #print coords
        #print "\n".join(repr(self.__dict__).split(","))
        font.draw_rect(*coords)


# Main parser functions
def handle_tokens(texgroup, env):
    """Scans the entire (tex)group to handle tokens. Tokens are other groups,
    commands, characters, kerns etc. Used recursively.
    
    """
    result = []
    # So we're sure that nothing changes the outer environment
    env = env.copy()
    while texgroup:
        item = texgroup.pop(0)
        #print texgroup, type(texgroup)
        #print env.face, type(item), repr(item)
        if isinstance(item, list):
            appendix = handle_tokens(item, env.copy())
        elif item in scripts:
            sub, sup, texgroup = handle_scripts(item, texgroup, env.copy())
            try:
                nuc = result.pop()
            except IndexError:
                nuc = Hbox([])
            appendix = Scripted(env.copy(), nuc=nuc, sub=sub, sup=sup)
        elif is_command(item):
            command = item.strip(esc_char)
            texgroup, env = handle_command(command, texgroup, env.copy(),
                                            allowsetters=True)
            continue
        elif isinstance(item, _textclass):
            if item == word_delim and env.mode == "mathmode":
                # Disregard space in mathmode
                continue
            uniindex = ord(item)
            appendix = handle_char(uniindex, env.copy())
        else:
            appendix = item

        # Checking if we should append a kern
        try:
            kern = get_kern(env, result[-1], appendix)
            if kern.advance != 0.0:
                result.append(kern)
        except IndexError:
            pass
        result.append(appendix)
    return Hbox(env.copy(),result)

def handle_command(command, texgroup, env, allowsetters=False):
    """Handles TeX commands that don't have backward propagation, and
    aren't setting anything in the environment.
    
    """
    # First we deal with setters - commands that change the
    # environment of the current group (scope)
    if command in setters:
        if not allowsetters:
            raise TexParseError("Seter not allowed here")
        if command in faces:
            env.face = command
        else:
            raise TexParseError("Unknown setter: %s%s"%(esc_char, command))
        return texgroup, env
    elif command == "frac":
        texgroup, args = get_args(command, texgroup, env.copy(), 2)
        num, den = args
        frac = Fraction(env=env, num=num, den=den)
        appendix = frac
    elif command in functions:
        _tex = "%srm %sthinspace %s"%(esc_char, esc_char, command)
        appendix = handle_tokens(parse_tex(_tex), env.copy())
    elif command in (" "):
        space = get_space(env)
        appendix = Kern(env, space.advance)
    elif command == "thinspace":
        #print command
        space = get_space(env)
        appendix = Kern(env, 1/2.*space.advance)
    elif command in tex2uni:
        uniindex = tex2uni[command]
        appendix = handle_char(uniindex, env.copy())
    else:
        #appendix = handle_tokens([r"\backslash"] + [
                        #~ c for c in command], env.copy())
        #appendix.env = env.copy()
        #print appendix
        raise TexParseError("Unknown command: " + esc_char + command)
    appendix = [appendix]
    appendix.extend(texgroup) 
    #print "App",appendix
    return appendix, env
    
def handle_scripts(firsttype, texgroup, env):
    sub = None
    sup = None
    env = env.copy()
    # The environment for the script elements
    _env = env.copy()
    _env.scriptdepth += 1
    firstscript = texgroup.pop(0)
    if firstscript in scripts:
        # An "_" or "^", immediately folowed by another "_" or "^"
        raise TexParseError("Missing { inserted. " + firsttype + firstscript)
    elif is_command(firstscript):
        command = firstscript.strip(esc_char)
        texgroup, _env = handle_command(command, texgroup, _env)
        firstscript = texgroup.pop(0)
    else:
        _tmp = handle_tokens([firstscript], _env)
        firstscript = _tmp.items.pop(0)
    if firsttype == "_":
        sub = firstscript
    else:
        sup = firstscript
    # Check if the next item is also a command for scripting
    try:
        second = texgroup[0]
    except IndexError:
        second = None
    if second in scripts:
        secondtype = texgroup.pop(0)
        if secondtype == firsttype:
            raise TexParseError("Double script: " + secondtype)
        try:
            secondscript = texgroup.pop(0)
        except IndexError:
            raise TexParseError("Empty script: " + secondtype)
        if secondscript in scripts:
            # An "_" or "^", immediately folowed by another "_" or "^"
            raise TexParseError("Missing { inserted. "\
                                            + secondtype + secondscript)
        elif is_command(secondscript):
            command = secondscript.strip(esc_char)
            texgroup, _env = handle_command(command, texgroup, _env)
            secondscript = texgroup.pop(0)
        else:
            _tmp = handle_tokens([secondscript], _env)
            secondscript = _tmp.items.pop(0)
        if secondtype == "_":
            sub = secondscript
        else:
            sup = secondscript
    # Check if the next item is also a command for scripting
    try:
        next = texgroup[0]
    except IndexError:
        next = None
    if next in scripts:
        raise TexParseError("Double script: " + next)
    return sub, sup, texgroup

def handle_char(uniindex, env):
    env = env.copy()
    char = unichr(uniindex)
    if not env.face:
        env.face = infer_face(env, char)
    return TexCharClass(env, char, uniindex=uniindex)

def get_args(command, texgroup, env, num_args):
    """Returns the arguments needed by a TeX command"""
    args = []
    i = 0
    while i < num_args:
        try:
            arg = texgroup.pop(0)
        except IndexError:
            msg = "%s is missing it's %d argument"%(command, i+1)
            raise TexParseError(msg)
        # We skip space
        if arg == " ":
            continue
        tmp = handle_tokens([arg], env.copy())
        arg = tmp.items.pop()
        args.append(arg)
        i += 1
    return texgroup, args


# Functions exported to backends
def math_parse_s_ft2font(s, dpi, fontsize, angle=0):
    """This function is called by the Agg backend"""
    s = s[1:-1]
    parsed = parse_tex(_textclass(s))
    env = environment.copy()
    env.dpi = dpi
    env.fontsize = fontsize
    parsed = handle_tokens(parsed, env)
    #print "\n".join(str(parsed.__dict__).split(","))
    width, height = parsed.width + 2, parsed.height + 2
    #print width, height
    for key in fonts:
        fonts[key].set_bitmap_size(width, height)
    parsed.render(-parsed.bearingx, height + parsed.ymin - 1)
    #~ parsed.render(-parsed.bearingx, height - 1 - (
                        #~ parsed.height - parsed.bearingy))
    _fonts = fonts.values()
    return width, height, _fonts

def math_parse_s_ft2font1(s, dpi, fontsize, angle=0):
    "Used only for testing"
    s = s[1:-1]
    parsed = parse_tex(_textclass(s))
    env = environment.copy()
    env.dpi = dpi
    env.fontsize = fontsize
    parsed = handle_tokens(parsed, env)
    #print "\n".join(str(parsed.__dict__).split(","))
    width, height = parsed.width + 10, parsed.height + 10
    width, height = 300, 300
    #print width, height
    for key in fonts:
        fonts[key].set_bitmap_size(width, height)
    parsed.render(width/2., height/2.)
    #fonts["mit"].draw_rect(0, 0, 40, 0)
    #fonts["mit"].draw_rect(0, 1, 40, 0)
    #parsed.render(20, 20)
    #~ parsed.render(-parsed.bearingx, height - 1 - (
                        #~ parsed.height - parsed.bearingy))
    _fonts = fonts.values()
    return width, height, _fonts


if __name__ == '__main__':
    #texstring = r"\\{ \horse\   Hello\^ ^ a^b_c}"
    #texstring = r"  asdf { \horse{}tralala1234\ \zztop{} \ Hello\^^a^{b_c}}"
    #texstring = r"{}{} { }"
    #texstring = r"{{{_ }}}"
    #texstring = r"\horse{}"
    #texstring = r"\horse;,.?)_)(*(*^*%&$$%{} Haha! Kako je frajeru?"
    #texstring = r"a_2\trav 32"
    #texstring = r"a_24{\sum_4^5} _3"
    texstring = _textclass(r"1_2^{4^5}32 5")
    parsed = parse_tex(texstring)
    #~ print bool(a)
    #print is_scriptcommand('\\subscript')