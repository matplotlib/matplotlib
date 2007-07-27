"""
"""

USE_NEW_CONFIG = True

from rcparams import rc
from cutils import get_config_file

if USE_NEW_CONFIG:
    from mplconfig import rcParams, mplConfig, save_config
else:
    from rcparams import rcParams
