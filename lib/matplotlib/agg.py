# This file was created automatically by SWIG.
# Don't modify this file, modify the SWIG interface instead.
# This file is compatible with both classic and new-style classes.

import _agg

def _swig_setattr_nondynamic(self,class_type,name,value,static=1):
    if (name == "this"):
        if isinstance(value, class_type):
            self.__dict__[name] = value.this
            if hasattr(value,"thisown"): self.__dict__["thisown"] = value.thisown
            del value.thisown
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    if (not static) or hasattr(self,name) or (name == "thisown"):
        self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)

def _swig_setattr(self,class_type,name,value):
    return _swig_setattr_nondynamic(self,class_type,name,value,0)

def _swig_getattr(self,class_type,name):
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError,name

import types
try:
    _object = types.ObjectType
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0
del types


cover_shift = _agg.cover_shift
cover_size = _agg.cover_size
cover_mask = _agg.cover_mask
cover_none = _agg.cover_none
cover_full = _agg.cover_full

deg2rad = _agg.deg2rad

rad2deg = _agg.rad2deg
path_cmd_stop = _agg.path_cmd_stop
path_cmd_move_to = _agg.path_cmd_move_to
path_cmd_line_to = _agg.path_cmd_line_to
path_cmd_curve3 = _agg.path_cmd_curve3
path_cmd_curve4 = _agg.path_cmd_curve4
path_cmd_end_poly = _agg.path_cmd_end_poly
path_cmd_mask = _agg.path_cmd_mask
path_flags_none = _agg.path_flags_none
path_flags_ccw = _agg.path_flags_ccw
path_flags_cw = _agg.path_flags_cw
path_flags_close = _agg.path_flags_close
path_flags_mask = _agg.path_flags_mask

is_vertex = _agg.is_vertex

is_stop = _agg.is_stop

is_move_to = _agg.is_move_to

is_line_to = _agg.is_line_to

is_curve = _agg.is_curve

is_curve3 = _agg.is_curve3

is_curve4 = _agg.is_curve4

is_end_poly = _agg.is_end_poly

is_close = _agg.is_close

is_next_poly = _agg.is_next_poly

is_cw = _agg.is_cw

is_ccw = _agg.is_ccw

is_oriented = _agg.is_oriented

is_closed = _agg.is_closed

get_close_flag = _agg.get_close_flag

clear_orientation = _agg.clear_orientation

get_orientation = _agg.get_orientation

set_orientation = _agg.set_orientation
class point_type(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, point_type, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, point_type, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ agg::point_type instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    __swig_setmethods__["x"] = _agg.point_type_x_set
    __swig_getmethods__["x"] = _agg.point_type_x_get
    if _newclass:x = property(_agg.point_type_x_get, _agg.point_type_x_set)
    __swig_setmethods__["y"] = _agg.point_type_y_set
    __swig_getmethods__["y"] = _agg.point_type_y_get
    if _newclass:y = property(_agg.point_type_y_get, _agg.point_type_y_set)
    def __init__(self, *args):
        _swig_setattr(self, point_type, 'this', _agg.new_point_type(*args))
        _swig_setattr(self, point_type, 'thisown', 1)
    def __del__(self, destroy=_agg.delete_point_type):
        try:
            if self.thisown: destroy(self)
        except: pass


class point_typePtr(point_type):
    def __init__(self, this):
        _swig_setattr(self, point_type, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, point_type, 'thisown', 0)
        _swig_setattr(self, point_type,self.__class__,point_type)
_agg.point_type_swigregister(point_typePtr)
cvar = _agg.cvar
pi = cvar.pi

class vertex_type(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, vertex_type, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, vertex_type, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ agg::vertex_type instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    __swig_setmethods__["x"] = _agg.vertex_type_x_set
    __swig_getmethods__["x"] = _agg.vertex_type_x_get
    if _newclass:x = property(_agg.vertex_type_x_get, _agg.vertex_type_x_set)
    __swig_setmethods__["y"] = _agg.vertex_type_y_set
    __swig_getmethods__["y"] = _agg.vertex_type_y_get
    if _newclass:y = property(_agg.vertex_type_y_get, _agg.vertex_type_y_set)
    __swig_setmethods__["cmd"] = _agg.vertex_type_cmd_set
    __swig_getmethods__["cmd"] = _agg.vertex_type_cmd_get
    if _newclass:cmd = property(_agg.vertex_type_cmd_get, _agg.vertex_type_cmd_set)
    def __init__(self, *args):
        _swig_setattr(self, vertex_type, 'this', _agg.new_vertex_type(*args))
        _swig_setattr(self, vertex_type, 'thisown', 1)
    def __del__(self, destroy=_agg.delete_vertex_type):
        try:
            if self.thisown: destroy(self)
        except: pass


class vertex_typePtr(vertex_type):
    def __init__(self, this):
        _swig_setattr(self, vertex_type, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, vertex_type, 'thisown', 0)
        _swig_setattr(self, vertex_type,self.__class__,vertex_type)
_agg.vertex_type_swigregister(vertex_typePtr)

class rect(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, rect, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, rect, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ agg::rect_base<int > instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    __swig_setmethods__["x1"] = _agg.rect_x1_set
    __swig_getmethods__["x1"] = _agg.rect_x1_get
    if _newclass:x1 = property(_agg.rect_x1_get, _agg.rect_x1_set)
    __swig_setmethods__["y1"] = _agg.rect_y1_set
    __swig_getmethods__["y1"] = _agg.rect_y1_get
    if _newclass:y1 = property(_agg.rect_y1_get, _agg.rect_y1_set)
    __swig_setmethods__["x2"] = _agg.rect_x2_set
    __swig_getmethods__["x2"] = _agg.rect_x2_get
    if _newclass:x2 = property(_agg.rect_x2_get, _agg.rect_x2_set)
    __swig_setmethods__["y2"] = _agg.rect_y2_set
    __swig_getmethods__["y2"] = _agg.rect_y2_get
    if _newclass:y2 = property(_agg.rect_y2_get, _agg.rect_y2_set)
    def __init__(self, *args):
        _swig_setattr(self, rect, 'this', _agg.new_rect(*args))
        _swig_setattr(self, rect, 'thisown', 1)
    def normalize(*args): return _agg.rect_normalize(*args)
    def clip(*args): return _agg.rect_clip(*args)
    def is_valid(*args): return _agg.rect_is_valid(*args)
    def __del__(self, destroy=_agg.delete_rect):
        try:
            if self.thisown: destroy(self)
        except: pass


class rectPtr(rect):
    def __init__(self, this):
        _swig_setattr(self, rect, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, rect, 'thisown', 0)
        _swig_setattr(self, rect,self.__class__,rect)
_agg.rect_swigregister(rectPtr)

class rect_d(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, rect_d, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, rect_d, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ agg::rect_base<double > instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    __swig_setmethods__["x1"] = _agg.rect_d_x1_set
    __swig_getmethods__["x1"] = _agg.rect_d_x1_get
    if _newclass:x1 = property(_agg.rect_d_x1_get, _agg.rect_d_x1_set)
    __swig_setmethods__["y1"] = _agg.rect_d_y1_set
    __swig_getmethods__["y1"] = _agg.rect_d_y1_get
    if _newclass:y1 = property(_agg.rect_d_y1_get, _agg.rect_d_y1_set)
    __swig_setmethods__["x2"] = _agg.rect_d_x2_set
    __swig_getmethods__["x2"] = _agg.rect_d_x2_get
    if _newclass:x2 = property(_agg.rect_d_x2_get, _agg.rect_d_x2_set)
    __swig_setmethods__["y2"] = _agg.rect_d_y2_set
    __swig_getmethods__["y2"] = _agg.rect_d_y2_get
    if _newclass:y2 = property(_agg.rect_d_y2_get, _agg.rect_d_y2_set)
    def __init__(self, *args):
        _swig_setattr(self, rect_d, 'this', _agg.new_rect_d(*args))
        _swig_setattr(self, rect_d, 'thisown', 1)
    def normalize(*args): return _agg.rect_d_normalize(*args)
    def clip(*args): return _agg.rect_d_clip(*args)
    def is_valid(*args): return _agg.rect_d_is_valid(*args)
    def __del__(self, destroy=_agg.delete_rect_d):
        try:
            if self.thisown: destroy(self)
        except: pass


class rect_dPtr(rect_d):
    def __init__(self, this):
        _swig_setattr(self, rect_d, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, rect_d, 'thisown', 0)
        _swig_setattr(self, rect_d,self.__class__,rect_d)
_agg.rect_d_swigregister(rect_dPtr)


unite_rectangles = _agg.unite_rectangles

unite_rectangles_d = _agg.unite_rectangles_d

intersect_rectangles = _agg.intersect_rectangles

intersect_rectangles_d = _agg.intersect_rectangles_d
class trans_affine(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, trans_affine, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, trans_affine, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ agg::trans_affine instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, trans_affine, 'this', _agg.new_trans_affine(*args))
        _swig_setattr(self, trans_affine, 'thisown', 1)
    def parl_to_parl(*args): return _agg.trans_affine_parl_to_parl(*args)
    def rect_to_parl(*args): return _agg.trans_affine_rect_to_parl(*args)
    def parl_to_rect(*args): return _agg.trans_affine_parl_to_rect(*args)
    def reset(*args): return _agg.trans_affine_reset(*args)
    def multiply(*args): return _agg.trans_affine_multiply(*args)
    def premultiply(*args): return _agg.trans_affine_premultiply(*args)
    def invert(*args): return _agg.trans_affine_invert(*args)
    def flip_x(*args): return _agg.trans_affine_flip_x(*args)
    def flip_y(*args): return _agg.trans_affine_flip_y(*args)
    def as_vec6(*args): return _agg.trans_affine_as_vec6(*args)
    def load_from(*args): return _agg.trans_affine_load_from(*args)
    def __imul__(*args): return _agg.trans_affine___imul__(*args)
    def __mul__(*args): return _agg.trans_affine___mul__(*args)
    def __invert__(*args): return _agg.trans_affine___invert__(*args)
    def __eq__(*args): return _agg.trans_affine___eq__(*args)
    def __ne__(*args): return _agg.trans_affine___ne__(*args)
    def transform(*args): return _agg.trans_affine_transform(*args)
    def inverse_transform(*args): return _agg.trans_affine_inverse_transform(*args)
    def determinant(*args): return _agg.trans_affine_determinant(*args)
    def scale(*args): return _agg.trans_affine_scale(*args)
    def is_identity(*args): return _agg.trans_affine_is_identity(*args)
    def is_equal(*args): return _agg.trans_affine_is_equal(*args)
    def get_rotation(*args): return _agg.trans_affine_get_rotation(*args)
    def get_translation(*args): return _agg.trans_affine_get_translation(*args)
    def get_scaling(*args): return _agg.trans_affine_get_scaling(*args)
    def __del__(self, destroy=_agg.delete_trans_affine):
        try:
            if self.thisown: destroy(self)
        except: pass


class trans_affinePtr(trans_affine):
    def __init__(self, this):
        _swig_setattr(self, trans_affine, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, trans_affine, 'thisown', 0)
        _swig_setattr(self, trans_affine,self.__class__,trans_affine)
_agg.trans_affine_swigregister(trans_affinePtr)

class trans_affine_rotation(trans_affine):
    __swig_setmethods__ = {}
    for _s in [trans_affine]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, trans_affine_rotation, name, value)
    __swig_getmethods__ = {}
    for _s in [trans_affine]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, trans_affine_rotation, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ agg::trans_affine_rotation instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, trans_affine_rotation, 'this', _agg.new_trans_affine_rotation(*args))
        _swig_setattr(self, trans_affine_rotation, 'thisown', 1)
    def __del__(self, destroy=_agg.delete_trans_affine_rotation):
        try:
            if self.thisown: destroy(self)
        except: pass


class trans_affine_rotationPtr(trans_affine_rotation):
    def __init__(self, this):
        _swig_setattr(self, trans_affine_rotation, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, trans_affine_rotation, 'thisown', 0)
        _swig_setattr(self, trans_affine_rotation,self.__class__,trans_affine_rotation)
_agg.trans_affine_rotation_swigregister(trans_affine_rotationPtr)

class trans_affine_scaling(trans_affine):
    __swig_setmethods__ = {}
    for _s in [trans_affine]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, trans_affine_scaling, name, value)
    __swig_getmethods__ = {}
    for _s in [trans_affine]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, trans_affine_scaling, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ agg::trans_affine_scaling instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, trans_affine_scaling, 'this', _agg.new_trans_affine_scaling(*args))
        _swig_setattr(self, trans_affine_scaling, 'thisown', 1)
    def __del__(self, destroy=_agg.delete_trans_affine_scaling):
        try:
            if self.thisown: destroy(self)
        except: pass


class trans_affine_scalingPtr(trans_affine_scaling):
    def __init__(self, this):
        _swig_setattr(self, trans_affine_scaling, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, trans_affine_scaling, 'thisown', 0)
        _swig_setattr(self, trans_affine_scaling,self.__class__,trans_affine_scaling)
_agg.trans_affine_scaling_swigregister(trans_affine_scalingPtr)

class trans_affine_translation(trans_affine):
    __swig_setmethods__ = {}
    for _s in [trans_affine]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, trans_affine_translation, name, value)
    __swig_getmethods__ = {}
    for _s in [trans_affine]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, trans_affine_translation, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ agg::trans_affine_translation instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, trans_affine_translation, 'this', _agg.new_trans_affine_translation(*args))
        _swig_setattr(self, trans_affine_translation, 'thisown', 1)
    def __del__(self, destroy=_agg.delete_trans_affine_translation):
        try:
            if self.thisown: destroy(self)
        except: pass


class trans_affine_translationPtr(trans_affine_translation):
    def __init__(self, this):
        _swig_setattr(self, trans_affine_translation, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, trans_affine_translation, 'thisown', 0)
        _swig_setattr(self, trans_affine_translation,self.__class__,trans_affine_translation)
_agg.trans_affine_translation_swigregister(trans_affine_translationPtr)

class trans_affine_skewing(trans_affine):
    __swig_setmethods__ = {}
    for _s in [trans_affine]: __swig_setmethods__.update(_s.__swig_setmethods__)
    __setattr__ = lambda self, name, value: _swig_setattr(self, trans_affine_skewing, name, value)
    __swig_getmethods__ = {}
    for _s in [trans_affine]: __swig_getmethods__.update(_s.__swig_getmethods__)
    __getattr__ = lambda self, name: _swig_getattr(self, trans_affine_skewing, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ agg::trans_affine_skewing instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __init__(self, *args):
        _swig_setattr(self, trans_affine_skewing, 'this', _agg.new_trans_affine_skewing(*args))
        _swig_setattr(self, trans_affine_skewing, 'thisown', 1)
    def __del__(self, destroy=_agg.delete_trans_affine_skewing):
        try:
            if self.thisown: destroy(self)
        except: pass


class trans_affine_skewingPtr(trans_affine_skewing):
    def __init__(self, this):
        _swig_setattr(self, trans_affine_skewing, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, trans_affine_skewing, 'thisown', 0)
        _swig_setattr(self, trans_affine_skewing,self.__class__,trans_affine_skewing)
_agg.trans_affine_skewing_swigregister(trans_affine_skewingPtr)

class path_storage(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, path_storage, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, path_storage, name)
    def __repr__(self):
        return "<%s.%s; proxy of C++ agg::path_storage instance at %s>" % (self.__class__.__module__, self.__class__.__name__, self.this,)
    def __del__(self, destroy=_agg.delete_path_storage):
        try:
            if self.thisown: destroy(self)
        except: pass

    def __init__(self, *args):
        _swig_setattr(self, path_storage, 'this', _agg.new_path_storage(*args))
        _swig_setattr(self, path_storage, 'thisown', 1)
    def remove_all(*args): return _agg.path_storage_remove_all(*args)
    def last_vertex(*args): return _agg.path_storage_last_vertex(*args)
    def prev_vertex(*args): return _agg.path_storage_prev_vertex(*args)
    def rel_to_abs(*args): return _agg.path_storage_rel_to_abs(*args)
    def move_to(*args): return _agg.path_storage_move_to(*args)
    def move_rel(*args): return _agg.path_storage_move_rel(*args)
    def line_to(*args): return _agg.path_storage_line_to(*args)
    def line_rel(*args): return _agg.path_storage_line_rel(*args)
    def arc_to(*args): return _agg.path_storage_arc_to(*args)
    def arc_rel(*args): return _agg.path_storage_arc_rel(*args)
    def curve3(*args): return _agg.path_storage_curve3(*args)
    def curve3_rel(*args): return _agg.path_storage_curve3_rel(*args)
    def curve4(*args): return _agg.path_storage_curve4(*args)
    def curve4_rel(*args): return _agg.path_storage_curve4_rel(*args)
    def end_poly(*args): return _agg.path_storage_end_poly(*args)
    def close_polygon(*args): return _agg.path_storage_close_polygon(*args)
    def add_poly(*args): return _agg.path_storage_add_poly(*args)
    def add_vertices(*args): return _agg.path_storage_add_vertices(*args)
    def start_new_path(*args): return _agg.path_storage_start_new_path(*args)
    def copy_from(*args): return _agg.path_storage_copy_from(*args)
    def total_vertices(*args): return _agg.path_storage_total_vertices(*args)
    def command(*args): return _agg.path_storage_command(*args)
    def rewind(*args): return _agg.path_storage_rewind(*args)
    def vertex(*args): return _agg.path_storage_vertex(*args)
    def arrange_orientations(*args): return _agg.path_storage_arrange_orientations(*args)
    def arrange_orientations_all_paths(*args): return _agg.path_storage_arrange_orientations_all_paths(*args)
    def flip_x(*args): return _agg.path_storage_flip_x(*args)
    def flip_y(*args): return _agg.path_storage_flip_y(*args)
    def add_vertex(*args): return _agg.path_storage_add_vertex(*args)
    def modify_vertex(*args): return _agg.path_storage_modify_vertex(*args)
    def modify_command(*args): return _agg.path_storage_modify_command(*args)

class path_storagePtr(path_storage):
    def __init__(self, this):
        _swig_setattr(self, path_storage, 'this', this)
        if not hasattr(self,"thisown"): _swig_setattr(self, path_storage, 'thisown', 0)
        _swig_setattr(self, path_storage,self.__class__,path_storage)
_agg.path_storage_swigregister(path_storagePtr)


