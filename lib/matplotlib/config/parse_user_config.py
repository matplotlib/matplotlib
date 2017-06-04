import os
import json

import six


LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ALIAS_MAPPING = os.path.join(LOCAL_DIR, 'config_alias_map.json')
ALIAS_MAPPING = {}


def update_config_from_dict_path(config_dict, dict_path, value):
    """Set value in a config dictionary using a path string.

    Parameters
    ----------
    config_dict : dict
        Configuration dictionary matching format expected by
        ``matplotlib.config.mpl_config.MPLConfig``.
    dict_path : str
        String with nested dictionary keys separated by a colon.
        For example, 'a:b:c' maps to the key ``some_dict['a']['b']['c']``.
    value : object
        Configuration value.
    """
    dict_keys = dict_path.split(':')
    key_to_set = dict_keys.pop()

    inner_dict = config_dict
    for key in dict_keys:
        if key not in inner_dict:
            inner_dict[key] = {}
        inner_dict = inner_dict[key]
    inner_dict[key_to_set] = value


def load_config_mapping(filename):
    """Return dictionary mapping config labels to config paths.
    """
    with open(filename) as f:
        config_mapping = json.load(f)
    return config_mapping


def update_alias_mapping(filename):
    """Update mappings from user-config aliases to config dict paths. """
    ALIAS_MAPPING.update(load_config_mapping(filename))

update_alias_mapping(DEFAULT_ALIAS_MAPPING)


def user_key_to_dict_paths(key):
    """Return config-dict paths from user-config alias.

    See also ``update_config_from_dict_path``.
    """
    return ALIAS_MAPPING[key]


def update_config_dict_from_user_config(config_dict, user_config):
    """Update internal configuration dict from user-config dict.
    """
    for user_key, value in six.iteritems(user_config):
        dict_paths = user_key_to_dict_paths(user_key)
        for path in dict_paths:
            update_config_from_dict_path(config_dict, path, value)
