import numpy as np
from typing import Optional
from typing import Union
from matplotlib import _api


def world_transformation(xmin: int, xmax:int,
                         ymin: int, ymax: int,
                         zmin: int, zmax: int, pb_aspect=Optional[np.typing.ArrayLike]) -> np.ndarray: ...



def _rotation_about_vector(v: np.typing.ArrayLike, angle: float) -> np.ndarray: ...


def _view_axes(E: np.ndarray, R: np.ndarray, V: np.ndarray, roll: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


def _view_transformation_uvw(u: np.ndarray, v: np.ndarray, w: np.ndarray, E:np.ndarray) -> np.ndarray: ...

def _persp_transformation(zfront: np.typing.ArrayLike, zback: np.typing.ArrayLike, focal_length: np.typing.ArrayLike) -> np.ndarray: ...



def _ortho_transformation(zfront, zback) -> np.ndarray: ...



def _proj_transform_vec(vec: np.typing.ArrayLike, M: np.typing.ArrayLike)  -> tuple[float, float, float]: ...


def _proj_transform_vectors(vecs: np.ndarray, M: np.ndarray) -> np.ndarray: ...


def _proj_transform_vec_clip(vec: np.typing.ArrayLike, M: np.ndarray, focal_length:np.typing.ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


def inv_transform(xs: np.typing.ArrayLike, ys: np.typing.ArrayLike, zs: np.typing.ArrayLike, invM: np.ndarray) -> tuple[float, float, float]: ...


def _vec_pad_ones(xs: np.typing.ArrayLike, ys: np.typing.ArrayLike, zs: Union[np.typing.ArrayLike, float] = 0) -> np.ndarray: ...


def proj_transform(xs: np.typing.ArrayLike, ys: np.typing.ArrayLike, zs: np.typing.ArrayLike, M:np.ndarray) -> tuple[float, float, float]: ...


@_api.deprecated("3.10")
def proj_transform_clip(xs: np.typing.ArrayLike, ys: np.typing.ArrayLike, zs: np.typing.ArrayLike, M:np.ndarray)  -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

def _proj_transform_clip(xs: np.typing.ArrayLike, ys: np.typing.ArrayLike, zs: np.typing.ArrayLike, M:np.ndarray, focal_length: np.typing.ArrayLike)  -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...

def _proj_points(points: np.typing.ArrayLike, M: np.ndarray) -> np.ndarray: ...


def _proj_trans_points(points: np.typing.ArrayLike, M: np.ndarray) -> tuple[float, float, float]: ...
