#include <pybind11/pybind11.h>
#include <pybind11/native_enum.h>
#include <pybind11/numpy.h>
#ifdef PYBIND11_HAS_SUBINTERPRETER_SUPPORT
#include <pybind11/subinterpreter.h>
#endif

#include <algorithm>

#include "_image_resample.h"
#include "py_array.h"
#include "py_converters.h"

namespace py = pybind11;
using namespace pybind11::literals;

using MeshArray = mpl::array<double, mpl::shape<-1, 2>>;
using InputImageArray = mpl::array<const uint8_t, mpl::ndim<3>>;
using OutputImageArray = mpl::array<uint8_t, mpl::shape<-1, -1, 3>>;

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


static MeshArray
_get_transform_mesh(const py::object& transform, const py::ssize_t *dims)
{
    /* TODO: Could we get away with float, rather than double, arrays here? */

    /* Given a non-affine transform object, create a mesh that maps
    every pixel center in the output image to the input image.  This is used
    as a lookup table during the actual resampling. */

    // If attribute doesn't exist, raises Python AttributeError
    auto inverse = transform.attr("inverted")();

    auto input_mesh_arr = MeshArray(dims[0] * dims[1]);
    auto p = input_mesh_arr.data();

    for (auto y = 0; y < dims[0]; ++y) {
        for (auto x = 0; x < dims[1]; ++x) {
            // The convention for the supplied transform is that pixel centers
            // are at 0.5, 1.5, 2.5, etc.
            *p++ = (double)x + 0.5;
            *p++ = (double)y + 0.5;
        }
    }

    auto output_mesh_obj = inverse.attr("transform")(input_mesh_arr);
    auto output_mesh_arr = MeshArray(output_mesh_obj);

    // An undersized mesh would be read out of bounds by the resampler.
    if (!output_mesh_arr.is_same_shape(input_mesh_arr)) {
        throw std::runtime_error(mpl::format(
            "Inverse transformed mesh array should have shape {} not {}",
            input_mesh_arr.shape_str(), output_mesh_arr.shape_str()));
    }

    return output_mesh_arr;
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
        throw std::invalid_argument(mpl::format(
            "3D input array must be RGBA with shape (M, N, 4), has trailing dimension of {}",
            input_array.shape(2)));
    }

    // Ensure input array is contiguous, regardless of dtype
    input_array = py::array::ensure(input_array, py::array::c_style);
    if (!input_array) {
        throw std::invalid_argument("Input array could not be made C-contiguous");
    }

    // Validate output array
    auto out_ndim = output_array.ndim();

    if (out_ndim != ndim) {
        throw std::invalid_argument(mpl::format(
            "Input ({}D) and output ({}D) arrays have different dimensionalities",
            ndim, out_ndim));
    }

    if (out_ndim == 3 && output_array.shape(2) != 4) {
        throw std::invalid_argument(mpl::format(
            "3D output array must be RGBA with shape (M, N, 4), has trailing dimension of {}",
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
    std::optional<MeshArray> transform_mesh;

    // Validate transform
    if (transform.is_none()) {
        params.is_affine = true;
    } else {
        // Raises Python AttributeError if no such attribute or TypeError if cast fails
        bool is_affine = py::cast<bool>(transform.attr("is_affine"));

        if (is_affine) {
            convert_trans_affine(transform, params.affine);
            params.is_affine = is_affine;
        } else {
            transform_mesh = _get_transform_mesh(transform, output_array.shape());
            params.transform_mesh = transform_mesh->data();
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


[[noreturn]] static void
raise_image_comparison_failure(std::string msg)
{
    auto exceptions = py::module_::import("matplotlib.testing.exceptions");
    auto ImageComparisonFailure = exceptions.attr("ImageComparisonFailure");
    py::set_error(ImageComparisonFailure, msg.c_str());
    throw py::error_already_set();
}


// This is used by matplotlib.testing.compare to calculate RMS and a difference image.
static py::tuple
calculate_rms_and_diff(const mpl::defer<InputImageArray> &expected_image_obj,
                       const mpl::defer<InputImageArray> &actual_image_obj)
{
    std::optional<InputImageArray> expected_image_arr_opt;
    std::optional<InputImageArray> actual_image_arr_opt;

    try {
        expected_image_arr_opt = expected_image_obj.value("expected_image");
        actual_image_arr_opt = actual_image_obj.value("actual_image");
    } catch (const py::value_error &e) {
        raise_image_comparison_failure(e.what());
    }

    auto expected_image_arr = expected_image_arr_opt.value();
    auto actual_image_arr = actual_image_arr_opt.value();

    auto height = expected_image_arr.shape(0);
    auto width = expected_image_arr.shape(1);
    auto depth = expected_image_arr.shape(2);

    if (depth != 3 && depth != 4) {
        raise_image_comparison_failure(mpl::format(
            "expected_image must be RGB or RGBA but has depth {}", depth));
    }

    if (!expected_image_arr.is_same_shape(actual_image_arr)) {
        raise_image_comparison_failure(mpl::format(
            "Image shapes do not match. expected_image is {}, actual_image is {}",
            expected_image_arr.shape_str(), actual_image_arr.shape_str()));
    }

    auto expected_image = expected_image_arr.view();
    auto actual_image = actual_image_arr.view();

    auto diff_image_arr = OutputImageArray(height, width);
    auto diff_image = diff_image_arr.view();

    double total = 0.0;
    for (auto i = 0; i < height; i++) {
        for (auto j = 0; j < width; j++) {
            for (auto k = 0; k < depth; k++) {
                auto pixel_diff = static_cast<double>(expected_image(i, j, k)) -
                                  static_cast<double>(actual_image(i, j, k));

                total += pixel_diff*pixel_diff;

                if (k != 3) { // Hard-code a fully solid alpha channel by omitting it.
                    diff_image(i, j, k) = static_cast<unsigned char>(std::clamp(
                        abs(pixel_diff) * 10, // Expand differences in luminance domain.
                        0.0, 255.0));
                }
            }
        }
    }
    total = total / (width * height * depth);

    return py::make_tuple(sqrt(total), diff_image_arr);
}


#ifdef PYBIND11_HAS_SUBINTERPRETER_SUPPORT
PYBIND11_MODULE(_image, m,
                py::mod_gil_not_used(), py::multiple_interpreters::per_interpreter_gil())
#else
PYBIND11_MODULE(_image, m, py::mod_gil_not_used())
#endif
{
    py::native_enum<interpolation_e>(m, "_InterpolationType", "enum.Enum")
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
        .export_values()
        .finalize();

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
