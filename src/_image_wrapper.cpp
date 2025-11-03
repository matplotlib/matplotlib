#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>

#include "_image_resample.h"
#include "py_converters.h"

namespace py = pybind11;
using namespace pybind11::literals;

/**********************************************************************
 * Free functions
 * */

const char* image_resample__doc__ =
R"""(Resample input_array, blending it in-place into output_array, using an affine transform.

Parameters
----------
input_array : 2-d or 3-d NumPy array of float, double or `numpy.uint8`
    If 2-d, the image is grayscale.  If 3-d, the image must be of size 4 in the last
    dimension and represents RGBA data.

output_array : 2-d or 3-d NumPy array of float, double or `numpy.uint8`
    The dtype and number of dimensions must match `input_array`.

transform : matplotlib.transforms.Transform instance
    The transformation from the input array to the output array.

interpolation : int, default: NEAREST
    The interpolation method.  Must be one of the following constants defined in this
    module:

      NEAREST, BILINEAR, BICUBIC, SPLINE16, SPLINE36, HANNING, HAMMING, HERMITE, KAISER,
      QUADRIC, CATROM, GAUSSIAN, BESSEL, MITCHELL, SINC, LANCZOS, BLACKMAN

resample : bool, optional
    When `True`, use a full resampling method.  When `False`, only resample when the
    output image is larger than the input image.

alpha : float, default: 1
    The transparency level, from 0 (transparent) to 1 (opaque).

norm : bool, default: False
    Whether to norm the interpolation function.

radius: float, default: 1
    The radius of the kernel, if method is SINC, LANCZOS or BLACKMAN.
)""";


static py::array_t<double>
_get_transform_mesh(const py::object& transform, const py::ssize_t *dims)
{
    /* TODO: Could we get away with float, rather than double, arrays here? */

    /* Given a non-affine transform object, create a mesh that maps
    every pixel center in the output image to the input image.  This is used
    as a lookup table during the actual resampling. */

    // If attribute doesn't exist, raises Python AttributeError
    auto inverse = transform.attr("inverted")();

    py::ssize_t mesh_dims[2] = {dims[0]*dims[1], 2};
    py::array_t<double> input_mesh(mesh_dims);
    auto p = input_mesh.mutable_data();

    for (auto y = 0; y < dims[0]; ++y) {
        for (auto x = 0; x < dims[1]; ++x) {
            // The convention for the supplied transform is that pixel centers
	    // are at 0.5, 1.5, 2.5, etc.
            *p++ = (double)x + 0.5;
            *p++ = (double)y + 0.5;
        }
    }

    auto output_mesh = inverse.attr("transform")(input_mesh);

    auto output_mesh_array =
        py::array_t<double, py::array::c_style | py::array::forcecast>(output_mesh);

    if (output_mesh_array.ndim() != 2) {
        throw std::runtime_error(
            "Inverse transformed mesh array should be 2D not {}D"_s.format(
                output_mesh_array.ndim()));
    }

    return output_mesh_array;
}


// Using generic py::array for input and output arrays rather than the more usual
// py::array_t<type> as this function supports multiple array dtypes.
static void
image_resample(py::array input_array,
               py::array& output_array,
               const py::object& transform,
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
            "3D input array must be RGBA with shape (M, N, 4), has trailing dimension of {}"_s.format(
                input_array.shape(2)));
    }

    // Ensure input array is contiguous, regardless of dtype
    input_array = py::array::ensure(input_array, py::array::c_style);

    // Validate output array
    auto out_ndim = output_array.ndim();

    if (out_ndim != ndim) {
        throw std::invalid_argument(
            "Input ({}D) and output ({}D) arrays have different dimensionalities"_s.format(
                ndim, out_ndim));
    }

    if (out_ndim == 3 && output_array.shape(2) != 4) {
        throw std::invalid_argument(
            "3D output array must be RGBA with shape (M, N, 4), has trailing dimension of {}"_s.format(
                output_array.shape(2)));
    }

    if (!output_array.dtype().is(dtype)) {
        throw std::invalid_argument("Input and output arrays have mismatched types");
    }

    if ((output_array.flags() & py::array::c_style) == 0) {
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
    py::array_t<double> transform_mesh;

    // Validate transform
    if (transform.is_none()) {
        params.is_affine = true;
    } else {
        // Raises Python AttributeError if no such attribute or TypeError if cast fails
        bool is_affine = py::cast<bool>(transform.attr("is_affine"));

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
                (dtype.equal(py::dtype::of<std::uint8_t>())) ? resample<agg::gray8> :
                (dtype.equal(py::dtype::of<std::int8_t>())) ? resample<agg::gray8> :
                (dtype.equal(py::dtype::of<std::uint16_t>())) ? resample<agg::gray16> :
                (dtype.equal(py::dtype::of<std::int16_t>())) ? resample<agg::gray16> :
                (dtype.equal(py::dtype::of<float>())) ? resample<agg::gray32> :
                (dtype.equal(py::dtype::of<double>())) ? resample<agg::gray64> :
                nullptr) : (
            // ndim == 3
                (dtype.equal(py::dtype::of<std::uint8_t>())) ? resample<agg::rgba8> :
                (dtype.equal(py::dtype::of<std::int8_t>())) ? resample<agg::rgba8> :
                (dtype.equal(py::dtype::of<std::uint16_t>())) ? resample<agg::rgba16> :
                (dtype.equal(py::dtype::of<std::int16_t>())) ? resample<agg::rgba16> :
                (dtype.equal(py::dtype::of<float>())) ? resample<agg::rgba32> :
                (dtype.equal(py::dtype::of<double>())) ? resample<agg::rgba64> :
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


// This is used by matplotlib.testing.compare to calculate RMS and a difference image.
static py::tuple
calculate_rms_and_diff(py::array_t<unsigned char> expected_image,
                       py::array_t<unsigned char> actual_image)
{
    for (const auto & [image, name] : {std::pair{expected_image, "Expected"},
                                       std::pair{actual_image, "Actual"}})
    {
        if (image.ndim() != 3) {
            auto exceptions = py::module_::import("matplotlib.testing.exceptions");
            auto ImageComparisonFailure = exceptions.attr("ImageComparisonFailure");
            py::set_error(
                ImageComparisonFailure,
                "{name} image must be 3-dimensional, but is {ndim}-dimensional"_s.format(
                    "name"_a=name, "ndim"_a=image.ndim()));
            throw py::error_already_set();
        }
    }

    auto height = expected_image.shape(0);
    auto width = expected_image.shape(1);
    auto depth = expected_image.shape(2);

    if (depth != 3 && depth != 4) {
        auto exceptions = py::module_::import("matplotlib.testing.exceptions");
        auto ImageComparisonFailure = exceptions.attr("ImageComparisonFailure");
        py::set_error(
            ImageComparisonFailure,
            "Image must be RGB or RGBA but has depth {depth}"_s.format(
                "depth"_a=depth));
        throw py::error_already_set();
    }

    if (height != actual_image.shape(0) || width != actual_image.shape(1) ||
            depth != actual_image.shape(2)) {
        auto exceptions = py::module_::import("matplotlib.testing.exceptions");
        auto ImageComparisonFailure = exceptions.attr("ImageComparisonFailure");
        py::set_error(
            ImageComparisonFailure,
            "Image sizes do not match expected size: {expected_image.shape} "_s
            "actual size {actual_image.shape}"_s.format(
                "expected_image"_a=expected_image, "actual_image"_a=actual_image));
        throw py::error_already_set();
    }
    auto expected = expected_image.unchecked<3>();
    auto actual = actual_image.unchecked<3>();

    py::ssize_t diff_dims[3] = {height, width, 3};
    py::array_t<unsigned char> diff_image(diff_dims);
    auto diff = diff_image.mutable_unchecked<3>();

    double total = 0.0;
    for (auto i = 0; i < height; i++) {
        for (auto j = 0; j < width; j++) {
            for (auto k = 0; k < depth; k++) {
                auto pixel_diff = static_cast<double>(expected(i, j, k)) -
                                  static_cast<double>(actual(i, j, k));

                total += pixel_diff*pixel_diff;

                if (k != 3) { // Hard-code a fully solid alpha channel by omitting it.
                    diff(i, j, k) = static_cast<unsigned char>(std::clamp(
                        abs(pixel_diff) * 10, // Expand differences in luminance domain.
                        0.0, 255.0));
                }
            }
        }
    }
    total = total / (width * height * depth);

    return py::make_tuple(sqrt(total), diff_image);
}


PYBIND11_MODULE(_image, m, py::mod_gil_not_used())
{
    py::enum_<interpolation_e>(m, "_InterpolationType")
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
        "input_array"_a,
        "output_array"_a,
        "transform"_a,
        "interpolation"_a = interpolation_e::NEAREST,
        "resample"_a = false,
        "alpha"_a = 1,
        "norm"_a = false,
        "radius"_a = 1,
        image_resample__doc__);

    m.def("calculate_rms_and_diff", &calculate_rms_and_diff,
          "expected_image"_a, "actual_image"_a);
}
