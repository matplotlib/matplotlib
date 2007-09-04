import os
import sys
import warnings

import checkdep
from cutils import get_config_file, get_data_path
from verbose import verbose
from rcsetup import defaultParams, validate_backend, validate_toolbar
from rcsetup import validate_cairo_format

_deprecated_map = {
    'text.fontstyle':   'font.style',
    'text.fontangle':   'font.style',
    'text.fontvariant': 'font.variant',
    'text.fontweight':  'font.weight',
    'text.fontsize':    'font.size',
    'tick.size' :       'tick.major.size',
    }


class RcParams(dict):
    
    """A dictionary object including validation
    
    validating functions are defined and associated with rc parameters in
    rcsetup.py
    """
    
    validate = dict([ (key, converter) for key, (default, converter) in \
                     defaultParams.iteritems() ])
    
    def __setitem__(self, key, val):
        try:
            if key in _deprecated_map.keys():
                alt = _deprecated_map[key]
                warnings.warn('%s is deprecated in matplotlibrc. Use %s \
instead.'% (key, alt))
                key = alt
            cval = self.validate[key](val)
            dict.__setitem__(self, key, cval)
        except KeyError:
            raise KeyError('%s is not a valid rc parameter.\
See rcParams.keys() for a list of valid parameters.'%key)


def rc_params(fail_on_error=False):
    'Return the default params updated from the values in the rc file'

    fname = get_config_file()
    if not os.path.exists(fname):
        message = 'could not find rc file; returning defaults'
        ret =  dict([ (key, tup[0]) for key, tup in defaultParams.items()])
        warnings.warn(message)
        return ret

    cnt = 0
    rc_temp = {}
    for line in file(fname):
        cnt += 1
        strippedline = line.split('#',1)[0].strip()
        if not strippedline: continue
        tup = strippedline.split(':',1)
        if len(tup) !=2:
            warnings.warn('Illegal line #%d\n\t%s\n\tin file "%s"'%\
                          (cnt, line, fname))
            continue
        key, val = tup
        key = key.strip()
        val = val.strip()
        if key in rc_temp:
            warnings.warn('Duplicate key in file "%s", line #%d'%(fname,cnt))
        rc_temp[key] = (val, line, cnt)

    ret = RcParams([ (key, default) for key, (default, converter) in \
                    defaultParams.iteritems() ])

    for key in ('verbose.level', 'verbose.fileo'):
        if key in rc_temp:
            val, line, cnt = rc_temp.pop(key)
            if fail_on_error:
                ret[key] = val # try to convert to proper type or raise
            else:
                try: ret[key] = val # try to convert to proper type or skip
                except Exception, msg:
                    warnings.warn('Bad val "%s" on line #%d\n\t"%s"\n\tin file \
"%s"\n\t%s' % (val, cnt, line, fname, msg))

    verbose.set_level(ret['verbose.level'])
    verbose.set_fileo(ret['verbose.fileo'])

    for key, (val, line, cnt) in rc_temp.iteritems():
        if defaultParams.has_key(key):
            if fail_on_error:
                ret[key] = val # try to convert to proper type or raise
            else:
                try: ret[key] = val # try to convert to proper type or skip
                except Exception, msg:
                    warnings.warn('Bad val "%s" on line #%d\n\t"%s"\n\tin file \
"%s"\n\t%s' % (val, cnt, line, fname, msg))
        else:
            print >> sys.stderr, """
Bad key "%s" on line %d in
%s.
You probably need to get an updated matplotlibrc file from
http://matplotlib.sf.net/matplotlibrc or from the matplotlib source
distribution""" % (key, cnt, fname)

    if ret['datapath'] is None:
        ret['datapath'] = get_data_path()

    verbose.report('loaded rc file %s'%fname)
    
    return ret


# this is the instance used by the matplotlib classes
rcParams = rc_params()

rcParamsDefault = RcParams([ (key, default) for key, (default, converter) in \
                    defaultParams.iteritems() ])

rcParams['ps.usedistiller'] = checkdep.ps_distiller(rcParams['ps.usedistiller'])
rcParams['text.usetex'] = checkdep.usetex(rcParams['text.usetex'])

def rc(group, **kwargs):
    """
    Set the current rc params.  Group is the grouping for the rc, eg
    for lines.linewidth the group is 'lines', for axes.facecolor, the
    group is 'axes', and so on.  Group may also be a list or tuple
    of group names, eg ('xtick','ytick').  kwargs is a list of
    attribute name/value pairs, eg

      rc('lines', linewidth=2, color='r')

    sets the current rc params and is equivalent to

      rcParams['lines.linewidth'] = 2
      rcParams['lines.color'] = 'r'

    The following aliases are available to save typing for interactive
    users
        'lw'  : 'linewidth'
        'ls'  : 'linestyle'
        'c'   : 'color'
        'fc'  : 'facecolor'
        'ec'  : 'edgecolor'
        'mew' : 'markeredgewidth'
        'aa'  : 'antialiased'

    Thus you could abbreviate the above rc command as

          rc('lines', lw=2, c='r')


    Note you can use python's kwargs dictionary facility to store
    dictionaries of default parameters.  Eg, you can customize the
    font rc as follows

      font = {'family' : 'monospace',
              'weight' : 'bold',
              'size'   : 'larger',
             }

      rc('font', **font)  # pass in the font dict as kwargs

    This enables you to easily switch between several configurations.
    Use rcdefaults to restore the default rc params after changes.
    """

    aliases = {
        'lw'  : 'linewidth',
        'ls'  : 'linestyle',
        'c'   : 'color',
        'fc'  : 'facecolor',
        'ec'  : 'edgecolor',
        'mew' : 'markeredgewidth',
        'aa'  : 'antialiased',
        }

    if is_string_like(group):
        group = (group,)
    for g in group:
        for k,v in kwargs.items():
            name = aliases.get(k) or k
            key = '%s.%s' % (g, name)
            if not rcParams.has_key(key):
                raise KeyError('Unrecognized key "%s" for group "%s" and name "%s"' %
                               (key, g, name))

            rcParams[key] = v

def rcdefaults():
    """
    Restore the default rc params - the ones that were created at
    matplotlib load time
    """
    rcParams.update(rcParamsDefault)
