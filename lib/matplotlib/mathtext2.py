r"""
Supported commands:
-------------------
 * _, ^, to any depth
 * commands for typesetting functions (\sin, \cos etc.),
 * commands for changing the current font (\rm, \cal etc.),
 * Space/kern commands "\ ", \thinspace

Small TO-DO's:
--------------
 * Display braces etc. \} not working (displaying wierd characters) etc.
 * better placing of sub/superscripts. F_1^1y_{1_{2_{3_{4}^3}}3}1_23
 * implement crampedness (or is it smth. else?). y_1 vs. y_1^1
 * add better italic correction. F^1
 * implement other space/kern commands

TO-DO's:
--------
 * \frac, \over, \above, \choose etc.
 * Implement classes for Line, Fraction etc
 * Change env to be a new class, not a dict.
 * Add support for other backends

"""
import os
from math import fabs

from matplotlib import get_data_path, rcParams
from matplotlib._mathtext_data import tex2uni
from matplotlib.ft2font import FT2Font, KERNING_DEFAULT

_path = get_data_path()
faces = ('mit', 'rm', 'tt', 'cal', 'nonascii')
filenamesd = dict(
                [(face, os.path.join(_path, rcParams['mathtext.'
                    + face])) for face in faces])
fonts = dict(
            [ (face, FT2Font(filenamesd[face])) for
                face in faces]
            )

esc_char = '\\'
# Grouping delimiters
begin_group_char = '{'
end_group_char = '}'
dec_delim = '.'
word_delim = ' '
scripts = ("_", "^")
functions = ("sin", "tan", "cos", "exp", "arctan", "arccos", "arcsin", "cot")
mathstyles = ["display", "text", "script", "scriptscript"]
modes = ["mathmode", "displaymathmode"]
setters = faces
# Maximum number of nestings (groups within groups)
max_depth = 10

# The topmost environment
environment = {
"mode": "mathmode",
"mathstyle" : "display",
"cramped" : False,
# We start with zero scriptdepth (should be incremented by a Scripted
# instance)
"scriptdepth" : 0, 
"face" : None,
"fontsize" : 12,
"dpi" : 100,
}

# _textclass can be unicode or str. Subclassed by TexCharClass
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


# Helper functions used by the parser
def parse_tex(texstring):
    texstring = normalize_tex(texstring)
    _parsed = to_list(texstring)
    #_parsed = Group(_parsed)
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

    # Removing the escaped space-like characters. Unescaped space in TeX is
    # not important
    # Replacing all space-like characters with a single space word_delim
    texstring = word_delim.join(texstring.split())
    texstring = texstring.replace(esc_char + word_delim, esc_char + 'space'
                                    + word_delim)

    #~ # Dealing with "syntactic sugar" goes here (i.e. '_', '^' etc.)
    #~ texstring = texstring.replace(esc_char + '_', esc_char + 'underscore' + word_delim)
    #~ i = texstring.find('_' + word_delim)
    #~ if i != -1:
        #~ raise TexParseError('Subscripting with space not allowed')
    #~ texstring = texstring.replace('_', esc_char + 'sub' + word_delim)

    #~ texstring = texstring.replace(esc_char + '^', esc_char + 'circumflex' + word_delim)
    #~ i = texstring.find('^' + word_delim)
    #~ if i != -1:
        #~ raise TexParseError('Superscripting with space not allowed')
    #~ texstring = texstring.replace('^', esc_char + 'sup' + word_delim)

    # Removing unnecessary white space
    texstring = word_delim.join(texstring.split())

    return texstring

#~ def check_valid(parsed):
    #~ # First we check if sub/superscripts are properly ordered
    #~ for i in xrange(0, len(parsed)/4*4, 4):
        #~ four = parsed[i:i+4]
        #~ if four.count(esc_char + "sup") > 1:
            #~ raise TexParseError("Double superscript %s"%four)
        #~ if four.count(esc_char + "sub") > 1:
            #~ raise TexParseError("Double subscript %s"%four)

def is_command(item):
    try:
        return (item.startswith(esc_char) and
                (item.strip(esc_char) not in setters))
    except AttributeError:
        return False

def is_setter(item):
    try:
        return (item.startswith(esc_char) and
                (item.strip(esc_char) in setters))
    except AttributeError:
        return False

def is_scriptcommand(s):
    return is_command(s) and (s.strip(esc_char) in scripts)


# Helper functions used by the renderer
def get_frac_bar_height(env):
    # TO-DO: Find a better way to calculate the height of the rule
    c = TexCharClass(".")
    c.env = env.copy()
    c._init_renderer()
    return c.height

def get_font(env, item):
    face = env['face']
    if not face:
        face = infer_face(env, item)
    #print face, repr(item)
    fontsize = env['fontsize'] * scriptfactors[env['scriptdepth']]
    dpi = env['dpi']
    font = fonts[face]

    font.set_size(fontsize, dpi)
    return font
    #~ font = FT2Font(filenamesd[face])
    #~ if fonts:
        #~ fonts[max(fonts.keys()) + 1] = font
    #~ else:
        #~ fonts[1] = font
def infer_face(env, item):
    if isinstance(item, _textclass):
        if item.isalpha():
            if env["mode"] == "mathmode" and item < "z":
                face = "mit"
            else:
                # TO-DO: Perhapsh change to 'rm'
                face = "nonascii"
        elif item.isdigit():
            face = "rm"
        elif ord(item) < 256:
            face = "rm"
        else:
            face = "nonascii"
    else:
        face = "nonascii"
    return face

def get_space(env):
    space = TexCharClass(" ")
    space.env = env.copy()
    if not space.env["face"]:
        space.env["face"] = "rm"
    space._init_renderer()
    return space

def get_kern(first, second):
    try:
        if first.env == second.env and\
            isinstance(first,TexCharClass) and isinstance(first, TexCharClass):
            font = get_font(first.env, first)
            advance = -font.get_kerning(first.uniindex, second.uniindex,
                                        KERNING_DEFAULT)/64.0
            return Kern(advance)
        else:
            return Kern(0)
    except AttributeError:
        return Kern(0)


# Classes used by the renderer
# Basically, all classes implement _init_renderer, that prepares them for
# rendering.
# TO-DO: Do the job of _init_renderer in the __init__ method
class Group(list):
    """A class that corresponds to a TeX group ({}). It is also used for
    rendering.
    """
    def __init__(self, texlist):
        self.initialized = False
        self.env = None
        list.__init__(self, texlist)

    def __getitem__(self, index):
        try:
            return list.__getitem__(self, index)
        except IndexError:
            return Group([])

    def pop(self, index=-1):
        try:
            return list.pop(self, index)
        except IndexError:
            return Group([])

    def _init_renderer(self):
        if self.initialized:
            return
        self.initialized = True

        self.xmin, self.ymin, self.xmax, self.ymax = (0.0,)*4
        (self.width, self.height, self.advance, self.bearingx,
            self.bearingy)= (0,)*5
        if not self:
            # empty group
            return
        for item in self:
            item._init_renderer()
            # The xmin remains the xmin of the first item
            self.xmax += item.advance
            
            self.bearingy = max((item.bearingy, self.bearingy))
            self.ymax = max((item.ymax, self.ymax))
            self.ymin = min((item.ymin, self.ymin))
            self.advance += item.advance
            previous = item
        # The bearingx of the whole group is the bearingx of the first
        # item.
        first = self[0]
        first._init_renderer()
        self.bearingx = first.bearingx
        self.xmin = first.xmin
        #print "\n".join(str(first.__dict__).split(","))
        last = self[-1]
        last._init_renderer()
        #print repr(last.__dict__)
        #self.xmax -= (last.advance - last.xmax)
        self.xmax = self.advance + fabs(last.advance - last.xmax)
        self.width = self.xmax - self.xmin
        self.height = self.ymax - self.ymin

    def render(self, x, y):
        #print "Group", self.width, self.height
        for item in self:
            #print "\n".join(str(item.__dict__).split(","))
            #print item.width, item.height
            item.render(x, y)
            x += item.advance


class Scripted:
    """Used for creating elements that have sub/superscripts"""
    def __init__(self, nuc=Group([]), type="ord", sub=Group([]),
        sup=Group([])):
        self.nuc = nuc
        self.type = type
        self.sub = sub
        self.sup = sup
        self.env = None
        self.initialized = False
    
    def __repr__(self):
        tmp = (repr(i) for i in [self.env, self.nuc, self.type,
            self.sub, self.sup])
        tmp = tuple(tmp)
        return "Scripted(env=%s,nuc=%s, type=%s, \
sub=%s, sup=%s)"%tmp

    def _init_renderer(self):
        #return
        if self.initialized:
            return
        self.initialized = True

        self.nuc._init_renderer()
        self.sub._init_renderer()
        self.sup._init_renderer()
        nuc, sub, sup = self.nuc, self.sub, self.sup

        # Heruistics for figuring out how much the subscripts origin has to be
        # below the origin of the nucleus (the descent of the letter "j").
        # TO-DO: Change with a better alternative. Not working: F_1^1y_1
        c = TexCharClass("j")
        c.env = nuc.env.copy()
        c._init_renderer()
        self.subpad = c.height - c.bearingy
        #self.subpad = 0.5*sub.height
        # Similar for the superscript
        C = TexCharClass("M")
        C.env = nuc.env.copy()
        C._init_renderer()
        self.suppad = max(nuc.height/2., C.ymax/2.)# - C.bearingy
        #~ print self.suppad
        #~ self.suppad = self.subpad
        #~ print self.suppad

        #self.advance = nuc.advance + max((sub.advance, sup.advance))
        self.advance = nuc.advance + max((sub.advance, sup.advance))

        self.xmin = nuc.xmin

        #~ if sub.advance > sup.advance:
            #~ corr = sub.advance - sub.xmax
        #~ else:
            #~ corr = sup.advance - sup.xmax
        self.xmax = self.advance# - corr

        if sub:
            self.ymin = min(nuc.ymin, -self.subpad + sub.ymin)
        else:
            self.ymin = nuc.ymin

        if sup:
            self.ymax = max((nuc.ymax, self.suppad + sup.bearingy))
        else:
            self.ymax = nuc.ymax

        # The bearing of the whole element is the bearing of the nucleus
        self.bearingx = nuc.bearingx
        # Heruistics. Feel free to change
        self.bearingy = self.ymax

        self.width = self.xmax - self.xmin
        self.height = self.ymax - self.ymin

    def render(self, x, y):
        #return
        nuc, sub, sup = self.nuc, self.sub, self.sup
        nx = x
        ny = y

        subx = x + nuc.advance# + sub.bearingx
        suby = y + self.subpad# - subfactor*self.env["fontsize"]

        supx = x + nuc.advance# + sup.bearingx
        supy = y - self.suppad# + 10#subfactor*self.env.fontsize

        self.nuc.render(nx, ny)
        self.sub.render(subx, suby)
        self.sup.render(supx, supy)

class Fraction:
    """A class for rendering a fraction."""
    def __init__(self, numer, denom):
        self.numer = numer
        self.denom = denom
        self.bar = None
        self.env = None

    def _init_renderer(self):
        self.numer._init_renderer()
        self.denom._init_renderer()
        num, den = self.numer, self.denom
        
        # TO-DO: Find a better way to implement the fraction bar
        self.pad = get_frac_bar_height(self.env)
        pad = self.pad
        self.width = max(num.width, den.width)
        self.bar = Line(self.width + 2*pad, pad)
        self.bar.env = self.env.copy()
        #~ self.bar.bearingx = pad
        #~ self.bar.advance = self.bar.width + 2*pad
        #~ self.bar.bearingy = pad + pad

        self.xmin = 0
        #self.xmax = self.bar.advance
        self.xmax = self.bar.width# + 2*pad

        self.ymin = -(2*pad + num.height)
        self.ymax = 2*pad + den.height

        self.width = self.xmax - self.xmin
        self.height = self.ymax - self.ymin
        
        #self.bearingx = pad
        self.bearingx = 0
        self.bearingy = self.ymax
        #self.advance = self.bar.advance
        self.advance = self.xmax

    def render(self, x, y):
        pad = self.pad
        #print self.bar.xmax, self.bar.xmin, self.bar.ymin, self.bar.ymax
        self.bar.render(x, y)

        nx = x - self.numer.bearingx + (self.width - self.numer.width)/2.
        ny = y - 2*pad + (self.numer.height - self.numer.ymax)
        self.numer.render(nx, ny)
        
        dx = x - self.denom.bearingx+ (self.width - self.denom.width)/2.
        dy = y + 2*pad + self.denom.bearingy
        self.denom.render(dx, dy)


# Primitives
class TexCharClass(_textclass):
    """A class that implements rendering of a single character."""
    def __init__(self, char):
        # You should pass the char as an int in the case when the TeX parser
        # hits a TeX symbol (i.e. \sum). You can get the int value of the TeX
        # symbol from the dict matplotlib._mathtext_data.tex2uni
        msg = "TexCharClass takes a single char or an int (Unicode index)"
        if isinstance(char, int):
            self.uniindex = char
            char = unichr(self.uniindex)
        elif isinstance(char, basestring):
            self.uniindex = ord(char)
        else:
            raise ValueError(msg)
        if len(char) != 1:
            raise ValueError(msg)
        self.initialized = False
        self.env = None
        self.glyph = None
        _textclass.__init__(self, char)
        
    def _init_renderer(self):
        if self.initialized:
            return
        self.initialized = True
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


class Kern:
    """Class that implements the rendering of a Kern."""
    def __init__(self, advance):
        self.xmin, self.ymin, self.xmax, self.ymax = (0.0,)*4
        self.width = advance
        self.height = 0
        self.advance = advance
        self.bearingx = 0
        self.bearingy = 0

    def _init_renderer(self):
        pass

    def render(self, x, y):
        pass

    def __repr__(self):
        return "Kern(%s)"%self.advance

class Line:
    """Class that implements the rendering of a line."""
    def __init__(self, width, height):
        self.xmin = 0
        self.ymin = -height/2.
        self.xmax = width
        self.ymax = height/2.

        self.width = width
        self.height = height
        self.advance = width
        self.bearingx = 0
        self.bearingy = self.ymax

    def _init_renderer(self):
        pass

    def render(self, x, y):
        font = get_font(self.env, self)
        coords = (x + self.xmin, y + self.ymin, x + self.xmax,
                                                        y + self.ymax)
        #print coords
        #print "\n".join(repr(self.__dict__).split(","))
        font.draw_rect(*coords)
        pass


# Main parser functions
def handle_tokens(texgroup, env):
    """Scans the entire (tex)group to handle tokens. Tokens are groups,
    commands, characters, kerns etc. Used recursively.
    
    """
    result = Group([])
    result.env = env
    # So we're sure that nothing changes the result's environment
    env = env.copy()
    #check_valid(texgroup)
    texgroup = Group(texgroup)
    #texgroup.env = env
    while texgroup:
        #print texgroup, type(texgroup)
        item = texgroup.pop(0)
        #print env["face"], type(item), repr(item)
        #print "Current item", item
        if isinstance(item, list):
            item = handle_tokens(item, env.copy())
            appendix = item
        # First we deal with scripts
        if item in scripts:
            scripted, texgroup = handle_scripts(item, texgroup, env.copy())
            scripted.nuc = result.pop()
            appendix = scripted
        elif is_setter(item):
            setter = item.strip(esc_char)
            env = handle_setter(setter, env)
            continue
        elif is_command(item):
            command = item.strip(esc_char)
            texgroup = handle_command(command, texgroup, env)
            continue
        elif isinstance(item, _textclass):
            #print item
            # Handling of characters
            if item == word_delim and env["mode"] == "mathmode":
                # Disregard space in mathmode
                continue
            item = TexCharClass(item)
            _env = env.copy()
            item.env = _env
            appendix = item
            #print item, item.env
            #print env
        else:
            appendix = item
        kern = get_kern(result[-1], appendix)
        if kern.advance != 0:
            result.append(kern)
        result.append(appendix)
    return result

def handle_command(command, texgroup, env):
    """Handles TeX commands that don't have backward propagation, and
    aren't setting anything in the environment.
    
    """
    # Now, we deal with normal commands - commands that change only
    # the elements after them.
    env = env.copy()
    if command == "frac":
        texgroup = handle_tokens(texgroup, env)
        num = texgroup.pop(0)
        #print num
        texgroup = handle_tokens(texgroup, env)
        den = texgroup.pop(0)
        #print den
        frac = Fraction(num, den)
        frac.env = env
        space = esc_char + "thinspace"
        appendix = [space, frac, space]
    elif command in functions:
        _tex = "%srm %sthinspace %s"%(esc_char, esc_char, command)
        appendix = parse_tex(_tex)
    elif command in ("space", " "):
        space = get_space(env)
        appendix = Kern(space.advance)
    elif command == "thinspace":
        #print command
        space = get_space(env)
        appendix = Kern(1/2.*space.advance)
    elif command in tex2uni:
        symbol = TexCharClass(unichr(tex2uni[command]))
        symbol.env = env.copy()
        appendix = symbol
    else:
        #appendix = handle_tokens([r"\backslash"] + [
                        #~ c for c in command], env.copy())
        #appendix.env = env.copy()
        #print appendix
        raise TexParseError("Unknown command: " + esc_char + command)
    appendix = [appendix]
    appendix.extend(texgroup) 
    #print "App",appendix
    return appendix

def handle_setter(setter, env):
    # First we deal with setters - commands that change the
    # environment of the current group (scope)
    if setter in faces:
        env["face"] = setter
        return env
    else:
        raise TexParseError("Unknown setter: %s%s"%(esc_char, setter))
    return env

def handle_scripts(scripttype, texgroup, env, prevtype=None,
                            prevscripted=None):
    # If prevscripted is the same type raise an error
    if prevtype == scripttype:
        raise TexParseError("Double script: " + scripttype)
    env = env.copy()
    # The environment for the script elements
    _env = env.copy()
    _env['scriptdepth'] += 1
    script = texgroup.pop(0)
    if script in scripts:
        # A "_" or "^", immediately folowed by another "_" or "^"
        raise TexParseError("Missing { inserted. " + scripttype + script)
    elif is_setter(script):
        # TeX doesn't allow setters (macros with \let in them) after
        # "_" or "^" (Try $1^1_\rm1^1_1$ in TeX)
        raise TexParseError("Missing { inserted. %s%s%s"%(
                                scripttype,esc_char,script))
    elif is_command(script):
        command = script.strip(esc_char)
        texgroup = handle_command(command, texgroup, env)
        script = texgroup.pop(0)
    else:
        #print repr(script), type(script)
        _tmp = handle_tokens([script], _env)
        script = _tmp.pop(0)
    if not prevtype:
        scripted = Scripted()
    else:
        scripted = prevscripted
    if scripttype == "_":
        scripted.sub = script
        scripted.sub.env = _env
    else:
        scripted.sup = script
        scripted.sub.env = _env
    # Check if the next item is also a command for scripting
    print type(texgroup)
    try:
        next = texgroup[0]
    except:
        next = None
    if next in scripts:
        next = texgroup.pop(0)
        if prevtype:
            # Three scripts in a row
            raise TexParseError("Too many scripts: %s,%s,%s"%(
                                    prevtype, scripttype, next))
        else:
            # The second script
            scripted, texgroup = handle_scripts(next, texgroup, env,
                                prevtype=scripttype, prevscripted=scripted)
    scripted.env = env
    return scripted, texgroup


# Functions exported to backends
def math_parse_s_ft2font(s, dpi, fontsize, angle=0):
    """This function is called by the Agg backend"""
    s = s[1:-1]
    parsed = parse_tex(_textclass(s))
    env = environment.copy()
    env["dpi"] = dpi
    env["fontsize"] = fontsize
    parsed = handle_tokens(parsed, env)
    parsed._init_renderer()
    #print "\n".join(str(parsed.__dict__).split(","))
    width, height = parsed.width + 2, parsed.height + 2
    print width, height
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
    env["dpi"] = dpi
    env["fontsize"] = fontsize
    parsed = handle_tokens(parsed, env)
    parsed._init_renderer()
    #print "\n".join(str(parsed.__dict__).split(","))
    width, height = parsed.width + 10, parsed.height + 10
    width, height = 300, 300
    print width, height
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
    #~ a = Group([])
    #~ a._init_renderer()
    #~ print bool(a)
    #print is_scriptcommand('\\subscript')