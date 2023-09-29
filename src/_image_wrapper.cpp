#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "_image_resample.h"
#include "py_converters_11.h"


/**********************************************************************
 * Free functions
 * */

const char* image_resample__doc__ =
"Resample input_array, blending it in-place into output_array, using an\n"
"affine transformation.\n\n"

"Parameters\n"
"----------\n"
"input_array : 2-d or 3-d NumPy array of float, double or `numpy.uint8`\n"
"    If 2-d, the image is grayscale.  If 3-d, the image must be of size\n"
"    4 in the last dimension and represents RGBA data.\n\n"

"output_array : 2-d or 3-d NumPy array of float, double or `numpy.uint8`\n"
"    The dtype and number of dimensions must match `input_array`.\n\n"

"transform : matplotlib.transforms.Transform instance\n"
"    The transformation from the input array to the output array.\n\n"

"interpolation : int, default: NEAREST\n"
"    The interpolation method.  Must be one of the following constants\n"
"    defined in this module:\n\n"

"      NEAREST, BILINEAR, BICUBIC, SPLINE16, SPLINE36,\n"
"      HANNING, HAMMING, HERMITE, KAISER, QUADRIC, CATROM, GAUSSIAN,\n"
"      BESSEL, MITCHELL, SINC, LANCZOS, BLACKMAN\n\n"

"resample : bool, optional\n"
"    When `True`, use a full resampling method.  When `False`, only\n"
"    resample when the output image is larger than the input image.\n\n"

"alpha : float, default: 1\n"
"    The transparency level, from 0 (transparent) to 1 (opaque).\n\n"

"norm : bool, default: False\n"
"    Whether to norm the interpolation function.\n\n"

"radius: float, default: 1\n"
"    The radius of the kernel, if method is SINC, LANCZOS or BLACKMAN.\n";


static pybind11::array_t<double>
_get_transform_mesh(const pybind11::object& transform, const pybind11::ssize_t *dims)
{
    /* TODO: Could we get away with float, rather than double, arrays here? */

    /* Given a non-affine transform object, create a mesh that maps
    every pixel in the output image to the input image.  This is used
    as a lookup table during the actual resampling. */

    // If attribute doesn't exist, raises Python AttributeError
    auto inverse = transform.attr("inverted")();

    pybind11::ssize_t mesh_dims[2] = {dims[0]*dims[1], 2};
    pybind11::array_t<double> input_mesh(mesh_dims);
    auto p = input_mesh.mutable_data();

    for (auto y = 0; y < dims[0]; ++y) {
        for (auto x = 0; x < dims[1]; ++x) {
            *p++ = (double)x;
            *p++ = (double)y;
        }
    }

    auto output_mesh = inverse.attr("transform")(input_mesh);

    auto output_mesh_array =
        pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>(output_mesh);

    if (output_mesh_array.ndim() != 2) {
        throw std::runtime_error(
            "Inverse transformed mesh array should be 2D not " +
            std::to_string(output_mesh_array.ndim()) + "D");
    }

    return output_mesh_array;
}


// Using generic pybind::array for input and output arrays rather than the more usual
// pybind::array_t<type> as function supports multiple array dtypes.
static void
image_resample(pybind11::array input_array,
               pybind11::array& output_array,
               const pybind11::object& transform,
               interpolation_e interpolation,
               bool resample_,  // Avoid name clash with resample() function
               float alpha,
               bool norm,
               float radius)
{
    // Validate input_array
    auto dtype = input_array.dtype();  // Validated when determine resampler below
    auto ndim = input_array.ndim();

    if (ndim != 2 && ndim != 3) {
        throw std::invalid_argument("Input array must be a 2D or 3D array");
    }

    if (ndim == 3 && input_array.shape(2) != 4) {
        throw std::invalid_argument(
            "3D input array must be RGBA with shape (M, N, 4), has trailing dimension of " +
            std::to_string(input_array.shape(2)));
    }

    // Ensure input array is contiguous, regardless of dtype
    input_array = pybind11::array::ensure(input_array, pybind11::array::c_style);

    // Validate output array
    auto out_ndim = output_array.ndim();

    if (out_ndim != ndim) {
        throw std::invalid_argument(
            "Input (" + std::to_string(ndim) + "D) and output (" + std::to_string(out_ndim) +
            "D) arrays have different dimensionalities");
    }

    if (out_ndim == 3 && output_array.shape(2) != 4) {
        throw std::invalid_argument(
            "3D output array must be RGBA with shape (M, N, 4), has trailing dimension of " +
            std::to_string(output_array.shape(2)));
    }

    if (!output_array.dtype().is(dtype)) {
        throw std::invalid_argument("Input and output arrays have mismatched types");
    }

    if ((output_array.flags() & pybind11::array::c_style) == 0) {
        throw std::invalid_argument("Output array must be C-contiguous");
    }

    if (!output_array.writeable()) {
        throw std::invalid_argument("Output array must be writeable");
    }

    resample_params_t params;
    params.interpolation = interpolation;
    params.transform_mesh = nullptr;
    params.resample = resample_;
    params.norm = norm;
    params.radius = radius;
    params.alpha = alpha;

    // Only used if transform is not affine.
    // Need to keep it in scope for the duration of this function.
    pybind11::array_t<double> transform_mesh;

    // Validate transform
    if (transform.is_none()) {
        params.is_affine = true;
    } else {
        // Raises Python AttributeError if no such attribute or TypeError if cast fails
        bool is_affine = pybind11::cast<bool>(transform.attr("is_affine"));

        if (is_affine) {
            convert_trans_affine(transform, params.affine);
            params.is_affine = true;
        } else {
            transform_mesh = _get_transform_mesh(transform, output_array.shape());
            params.transform_mesh = transform_mesh.data();
            params.is_affine = false;
        }
    }

    if (auto resampler =
            (ndim == 2) ? (
                (dtype.is(pybind11::dtype::of<std::uint8_t>())) ? resample<agg::gray8> :
                (dtype.is(pybind11::dtype::of<std::int8_t>())) ? resample<agg::gray8> :
                (dtype.is(pybind11::dtype::of<std::uint16_t>())) ? resample<agg::gray16> :
                (dtype.is(pybind11::dtype::of<std::int16_t>())) ? resample<agg::gray16> :
                (dtype.is(pybind11::dtype::of<float>())) ? resample<agg::gray32> :
                (dtype.is(pybind11::dtype::of<double>())) ? resample<agg::gray64> :
                nullptr) : (
            // ndim == 3
                (dtype.is(pybind11::dtype::of<std::uint8_t>())) ? resample<agg::rgba8> :
                (dtype.is(pybind11::dtype::of<std::int8_t>())) ? resample<agg::rgba8> :
                (dtype.is(pybind11::dtype::of<std::uint16_t>())) ? resample<agg::rgba16> :
                (dtype.is(pybind11::dtype::of<std::int16_t>())) ? resample<agg::rgba16> :
                (dtype.is(pybind11::dtype::of<float>())) ? resample<agg::rgba32> :
                (dtype.is(pybind11::dtype::of<double>())) ? resample<agg::rgba64> :
                nullptr)) {
        Py_BEGIN_ALLOW_THREADS
        resampler(
            input_array.data(), input_array.shape(1), input_array.shape(0),
            output_array.mutable_data(), output_array.shape(1), output_array.shape(0),
            params);
        Py_END_ALLOW_THREADS
    } else {
        throw std::invalid_argument("arrays must be of dtype byte, short, float32 or float64");
    }
}


PYBIND11_MODULE(_image, m) {
    pybind11::enum_<interpolation_e>(m, "_InterpolationType")
        .value("NEAREST", NEAREST)
        .value("BILINEAR", BILINEAR)
        .value("BICUBIC", BICUBIC)
        .value("SPLINE16", SPLINE16)
        .value("SPLINE36", SPLINE36)
        .value("HANNING", HANNING)
        .value("HAMMING", HAMMING)
        .value("HERMITE", HERMITE)
        .value("KAISER", KAISER)
        .value("QUADRIC", QUADRIC)
        .value("CATROM", CATROM)
        .value("GAUSSIAN", GAUSSIAN)
        .value("BESSEL", BESSEL)
        .value("MITCHELL", MITCHELL)
        .value("SINC", SINC)
        .value("LANCZOS", LANCZOS)
        .value("BLACKMAN", BLACKMAN)
        .export_values();

    m.def("resample", &image_resample,
        pybind11::arg("input_array"),
        pybind11::arg("output_array"),
        pybind11::arg("transform"),
        pybind11::arg("interpolation") = interpolation_e::NEAREST,
        pybind11::arg("resample") = false,
        pybind11::arg("alpha") = 1,
        pybind11::arg("norm") = false,
        pybind11::arg("radius") = 1,
        image_resample__doc__);
}
