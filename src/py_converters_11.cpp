#include "py_converters_11.h"

void convert_trans_affine(const py::object& transform, agg::trans_affine& affine)
{
    // If None assume identity transform so leave affine unchanged
    if (transform.is_none()) {
        return;
    }

    auto array = py::array_t<double, py::array::c_style>::ensure(transform);
    if (!array || array.ndim() != 2 || array.shape(0) != 3 || array.shape(1) != 3) {
        throw std::invalid_argument("Invalid affine transformation matrix");
    }

    auto buffer = array.data();
    affine.sx = buffer[0];
    affine.shx = buffer[1];
    affine.tx = buffer[2];
    affine.shy = buffer[3];
    affine.sy = buffer[4];
    affine.ty = buffer[5];
}
