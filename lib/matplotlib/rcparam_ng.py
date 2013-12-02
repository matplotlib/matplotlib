from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
from collections import namedtuple
from copy import copy

from functools import wraps

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
    @wraps(orig_fun)
    def wrapper(*args, **kwargs):
        for k, v in new_defaults.iteritems():
            if k not in kwargs:
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
