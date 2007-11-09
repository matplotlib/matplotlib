import os
import re
import sys
import warnings
import distutils.version as version

tex_req = '3.1415'
gs_req = '7.07'
gs_sugg = '8.60'
dvipng_req = '1.5'
pdftops_req = '3.0'

def dvipng():
    try:
        stdin, stdout = os.popen4('dvipng -version')
        return stdout.readlines()[1].split()[-1]
    except (IndexError, ValueError):
        return None

def ghostscript():
    try:
        if sys.platform == 'win32':
            command = 'gswin32c --version'
        else:
            command = 'gs --version'
        stdin, stdout = os.popen4(command)
        return stdout.read()[:-1]
    except (IndexError, ValueError):
        return None

def tex():
    try:
        stdin, stdout = os.popen4('tex -version')
        line = stdout.readlines()[0]
        pattern = '3\.1\d+'
        match = re.search(pattern, line)
        return match.group(0)
    except (IndexError, ValueError, AttributeError):
        return None

def pdftops():
    try:
        stdin, stdout = os.popen4('pdftops -v')
        for line in stdout.readlines():
            if 'version' in line:
                return line.split()[-1]
    except (IndexError, ValueError):
        return None

def compare_versions(a, b):
    "return True if a is greater than or equal to b"
    if a:
        a = version.LooseVersion(a)
        b = version.LooseVersion(b)
        if a>=b: return True
        else: return False
    else: return False

def ps_distiller(s):
    if not s:
        return False

    flag = True
    gs_v = ghostscript()
    if compare_versions(gs_v, gs_sugg): pass
    elif compare_versions(gs_v, gs_req):
        verbose.report(('ghostscript-%s found. ghostscript-%s or later '
                        'is recommended to use the ps.usedistiller option.') %\
                        (gs_v, gs_sugg))
    else:
        flag = False
        warnings.warn(('matplotlibrc ps.usedistiller option can not be used '
                       'unless ghostscript-%s or later is installed on your '
                       'system.') % gs_req)

    if s == 'xpdf':
        pdftops_v = pdftops()
        if compare_versions(pdftops_v, pdftops_req): pass
        else:
            flag = False
            warnings.warn(('matplotlibrc ps.usedistiller can not be set to '
                           'xpdf unless pdftops-%s or later is installed on '
                           'your system.') % pdftops_req)

    if flag:
        return s
    else:
        return False

def usetex(s):
    if not s:
        return False

    flag = True

    tex_v = tex()
    if compare_versions(tex_v, tex_req): pass
    else:
        flag = False
        warnings.warn(('matplotlibrc text.usetex option can not be used '
                       'unless TeX-%s or later is '
                       'installed on your system') % tex_req)

    dvipng_v = dvipng()
    if compare_versions(dvipng_v, dvipng_req): pass
    else:
        flag = False
        warnings.warn( 'matplotlibrc text.usetex can not be used with *Agg '
                       'backend unless dvipng-1.5 or later is '
                       'installed on your system')

    gs_v = ghostscript()
    if compare_versions(gs_v, gs_sugg): pass
    elif compare_versions(gs_v, gs_req):
        verbose.report(('ghostscript-%s found. ghostscript-%s or later is '
                        'recommended for use with the text.usetex '
                        'option.') % (gs_v, gs_sugg))
    else:
        flag = False
        warnings.warn(('matplotlibrc text.usetex can not be used '
                       'unless ghostscript-%s or later is '
                       'installed on your system') % gs_req)

    return flag
