# External imports
import enthought.traits.api as T

# Matplotlib-specific imports
import cutils

# For the fontconfig pattern parser
from matplotlib.fontconfig_pattern import parse_fontconfig_pattern

# stolen from cbook
def is_string_like(obj):
    if hasattr(obj, 'shape'): return 0 # this is a workaround
                                       # for a bug in numeric<23.1
    try: obj + ''
    except (TypeError, ValueError): return 0
    return 1


##############################################################################
# Handlers and other support functions and data structures
##############################################################################


class BackendHandler(T.TraitHandler):
    """
    """
    backends = {'agg': 'Agg',
                'cairo': 'Cairo',
                'cocoaagg': 'CocoaAgg',
                'fltkagg': 'FltkAgg',
                'gtk': 'GTK',
                'gtkagg': 'GTKAgg',
                'gtkcairo': 'GTKCairo',
                'pdf': 'Pdf',
                'ps': 'PS',
                'qt4agg': 'Qt4Agg',
                'qtagg': 'QtAgg',
                'svg': 'SVG',
                'template': 'Template',
                'tkagg': 'TkAgg',
                'wx': 'WX',
                'wxagg': 'WXAgg'}

    def validate(self, object, name, value):
        try:
            return self.backends[value.lower()]
        except:
            return self.error(object, name, value)

    def info(self):
        be = self.backends.keys()
        be.sort
        return "one of: %s"% ' | '.join(['%s'%i for i in be])


class BoolHandler(T.TraitHandler):
    """
    """

    bools = {'true': True,
             'yes': True,
             'y' : True,
             'on': True,
             1: True,
             True: True,
             'false': False,
             'no': False,
             'n': False,
             'off': False,
             0: False,
             False: False}
       
    def validate(self, object, name, value):
        try:
            return self.bools[value]
        except:
            return self.error(object, name, value)

    def info(self):
        return "one of: %s"% ' | '.join(['%s'%i for i in self.bools.keys()])

flexible_true = T.Trait(True, BoolHandler())
flexible_false = T.Trait(False, BoolHandler())

class ColorHandler(T.TraitHandler):
    """
    """
       
    def validate(self, object, name, value):
        s=value
        if s.lower() == 'none': return 'None'
        if len(s)==1 and s.isalpha(): return s
        if s.find(',')>=0: # looks like an rgb
            # get rid of grouping symbols
            s = ''.join([ c for c in s if c.isdigit() or c=='.' or c==','])
            vals = s.split(',')
            if len(vals)!=3:
                raise ValueError('Color tuples must be length 3')

            try: return [float(val) for val in vals]
            except ValueError:
                raise ValueError('Could not convert all entries "%s" to floats'%s)

        if s.replace('.', '').isdigit(): # looks like scalar (grayscale)
            return s

        if len(s)==6 and s.isalnum(): # looks like hex
            return '#' + s
    
        if len(s)==7 and s.startswith('#') and s[1:].isalnum():
            return s

        if s.isalpha():
            #assuming a color name, hold on
            return s

        raise ValueError('%s does not look like color arg'%s)

    def info(self):
        return """\
any valid matplotlib color, eg an abbreviation like 'r' for red, a full
name like 'orange', a hex color like '#efefef', a grayscale intensity
like '0.5', or an RGBA tuple (1,0,0,1)"""


class IsWritableDir(T.TraitHandler):
    def validate(self, object, name, value):
        if cutils.is_writable_dir(value):
            return value
        else:
            raise OSError('%s is not a writable directory'%value)


colormaps = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn',
             'BuGn_r', 'BuPu', 'BuPu_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 
             'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 
             'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 
             'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG',
             'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r',
             'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy',
             'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 
             'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 
             'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'YlGn', 'YlGnBu', 
             'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 
             'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 
             'cool', 'cool_r', 'copper', 'copper_r', 'flag', 
             'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r',
             'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 
             'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 
             'gist_yarg', 'gist_yarg_r', 'gray', 'gray_r', 'hot', 'hot_r', 
             'hsv', 'hsv_r', 'jet', 'jet_r', 'pink', 'pink_r', 
             'prism', 'prism_r', 'spectral', 'spectral_r', 'spring', 
             'spring_r', 'summer', 'summer_r', 'winter', 'winter_r']

class FontconfigPatternHandler(T.TraitHandler):
    """This validates a fontconfig pattern by using the FontconfigPatternParser.
    The results are not actually used here, since we can't instantiate a
    FontProperties object at config-parse time."""
    def validate(self, object, name, value):
        # This will throw a ValueError if value does not parse correctly
        parse_fontconfig_pattern(value)
        return value

    def info(self):
        return """A fontconfig pattern.  See the fontconfig user manual for more information."""
