from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
from collections import namedtuple, defaultdict
from copy import copy
import inspect
import json

from functools import wraps

import matplotlib
from .parse_user_config import update_config_dict_from_user_config


_kw_dict_nm = '_kw_defaults'
_kw_entry = namedtuple('_kw_entry', ['orig_funtion', 'kw_dict'])


def set_defaults(cls, key, new_defaults):
    """
    Set a set of default kwargs for the function `key` on
    the class `cls`.

    If there are currently defaults set, they will be removed
    before `new_defaults` are set.

    Parameters
    ----------
    cls : class
        The class that `key` is a member function on

    key : str
       name of the function to set the default values for

    new_defaults : dict
       kwargs to set as the default
    """
    # if the class doesn't have this key, raise an exception
    if not hasattr(cls, key):
        raise ValueError(("The class {cls} does not have attribute" +
                         "{key}").format(cls=cls, key=key))

    # make sure the class has the persistent structure
    # saving the original function
    if not hasattr(cls, _kw_dict_nm):
        setattr(cls, _kw_dict_nm, dict())

    if not six.callable(getattr(cls, key)):
        raise ValueError("The attribute {key} of {cls} ".format(key=key,
                                                                cls=cls) +
                         "is not callable")

    kw_dict = getattr(cls, _kw_dict_nm)

    if key in kw_dict:
        orig_fun, old_dict = kw_dict.pop(key)
    else:
        orig_fun = getattr(cls, key)

    # make a copy of the input so we don't have to worry about side effects
    # or external changes
    new_defaults = copy(new_defaults)

    # make dictionary entry and shove into the dictionary
    kw_dict[key] = _kw_entry(orig_fun, new_defaults)

    # make wrapper function, closes over the copied dictionary
    # and original function
    @wraps(orig_fun)
    def wrapper(*args, **kwargs):
        for k, v in new_defaults.iteritems():
            if k not in kwargs or kwargs[k] is None:
                kwargs[k] = v
        return orig_fun(*args, **kwargs)

    setattr(cls, key, wrapper)


def update_defaults(cls, key, new_defaults):
    """
    Updates the default values set for the `key` method
    of the class `cls`.

    If no default values are currently set, set defaults
    to `new_defaults`, if there are currently defaults set
    update with the values in `new_defaults`

    Parameters
    ----------
    cls : class
        The class that `key` is a member function on

    key : str
       name of the function to set the default values for

    new_defaults : dict
       kwargs to set as the default
    """
    # if the class doesn't have this key, raise an exception
    if not hasattr(cls, key):
        raise ValueError(("The class {cls} does not have attribute" +
                         "{key}").format(cls=cls, key=key))

    # if there isn't the persistent structure, then no default is
    # set, call `set_defaults` and return
    if not hasattr(cls, _kw_dict_nm):
        set_defaults(cls, key, new_defaults)
        return
    # grab the persistent dict
    kw_dict = getattr(cls, _kw_dict_nm)
    # if key in the persistent structure
    if key in kw_dict:
        # grab the existing dict
        orig_fun, old_dict = kw_dict[key]
        # update it
        old_dict.update(new_defaults)
    else:
        # otherwise, pass on to `set_defaults` and return
        set_defaults(cls, key, new_defaults)
        return


def reset_defaults(cls, key):
    """
    Removes any set defaults from the function `key` on
    the class `cls`.

    Parameters
    ----------
    cls : class
        The class that `key` is a member function on

    key : str
       name of the function to set the default values for
    """
    # if the class doesn't have this key, raise an exception
    if not hasattr(cls, key):
        raise ValueError(("The class {cls} does not have attribute" +
                         "{key}").format(cls=cls, key=key))

    # if there isn't the persistent structure, then no default to
    # reset, return doing nothing
    if not hasattr(cls, _kw_dict_nm):
        return
    #
    kw_dict = getattr(cls, _kw_dict_nm)

    if key in kw_dict:
        # grab the original function
        orig_fun, old_dict = kw_dict.pop(key)
        # reset to the original function
        setattr(cls, key, orig_fun)


def raise_invalid_class_path_error(class_parts):
    class_path = '.'.join(class_parts)
    raise ValueError("Invalid class: %s" % class_path)


def string_to_class(klass):
    """
    Turns a string -> a class object
    """
    last_level = matplotlib
    # split the string
    split_klass = klass.split('.')
    # strip the matplotlib off the front
    if split_klass[0] == 'matplotlib':
        split_klass.pop(0)

    for _k in split_klass:
        if not hasattr(last_level, _k):
            raise_invalid_class_path_error(split_klass)
        last_level = getattr(last_level, _k)

    if not inspect.isclass(last_level):
        raise_invalid_class_path_error(split_klass)

    return last_level


class MPLConfig(object):
    """
    A class for keeping track of default values
    """
    def __init__(self, input_dict=None):
        """
        Parameters
        ----------
        input_dict : dict
            a dict of dicts.  Top level keys are strings from hte classes
            inner keys are function names, inner values are kwarg diccts

        """
        self.core_dict = defaultdict(dict)
        if input_dict is not None:
            self.core_dict.update(input_dict)

    def store_default(self, klass, key, new_defaults):
        """
        Adds an entry to the core for the given values

        Parameters
        ----------
        klass : str
            string name of class to set defaults for

        key : str
            function to set the defaults for

        new_defaults : dict
            dict containing the new defaults (kwarg pairs)
        """
        self.core_dict[klass][key] = new_defaults

    def set_defaults(self):
        """
        Set the defaults contained in this object.  Use `set_defaults`
        which removes any existing defaults, leaving only the values
        in this object in place.
        """
        # loop over the core dictionary
        for klass, kw_pair in six.iteritems(self.core_dict):
            # turn the string into a class
            cls = string_to_class(klass)
            # look over the list of keys and set the defaults
            for key, default_dict in six.iteritems(kw_pair):
                set_defaults(cls, key, default_dict)

    def update_defaults(self):
        """
        Update to the default values  contained in this object.
        Use `update_defaults` which leaves non-conflicting defaults
        in place.
        """
        # loop over the core dictionary
        for klass, kw_pair in six.iteritems(self.core_dict):
            # turn the string into a class
            cls = string_to_class(klass)
            # look over the list of keys and set the defaults
            for key, default_dict in six.iteritems(kw_pair):
                update_defaults(cls, key, default_dict)

    def to_json(self, out_file_path):
        """
        Dumps default values to json file.  Use `from_json` to
        recover.

        Parameters
        ----------
        out_file_path : str
            A valid path to save the json file to.  Will overwrite
            any existing file at path
        """
        with open(out_file_path, 'w') as fout:
            json.dump(self.core_dict, fout, ensure_ascii=False,
                      indent=4)

    @classmethod
    def from_json(cls, in_file_path):
        """
        Creates a new MPLConfig object from a json file.  (see
        `to_json` to dump to json).

        Parameters
        ----------
        in_file_path : str
            Path to a json file to load.
        """
        with open(in_file_path, 'r') as fin:
            in_dict = json.load(fin)
        return cls(in_dict)

    @classmethod
    def from_user_config(cls, user_config):
        config_dict = {}
        update_config_dict_from_user_config(config_dict, user_config)
        return cls(config_dict)
