#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>

#include "_image_resample.h"
#include "agg_convert_overloads.h"
#include "mpl_utils.h"
#include "mpl_array.h"
#include "mpl_shape.h"

namespace nb = nanobind;
using namespace nanobind::literals;

using MeshArray = mpl::array<double, mpl::shape<-1, 2>>;
using InputImageArray = mpl::array<const std::uint8_t, mpl::ndim<3>>;
using OutputImageArray = mpl::array<std::uint8_t, mpl::shape<-1, -1, 3>>;

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
_get_transform_mesh(const nb::object& transform, size_t height, size_t width)
{
    /* TODO: Could we get away with float, rather than double, arrays here? */

    /* Given a non-affine transform object, create a mesh that maps
    every pixel center in the output image to the input image.  This is used
    as a lookup table during the actual resampling. */

    // If attribute doesn't exist, raises Python AttributeError
    auto inverse = transform.attr("inverted")();

    auto input_mesh_arr = MeshArray(height * width);
    auto p = input_mesh_arr.data();

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
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


// Using generic nb::ndarray for input and output arrays rather than
// mpl::array as this function supports multiple array dtypes.
static void
image_resample(const nb::object& input_array_obj,
               const nb::object& output_array_obj,
               const nb::object& transform,
               interpolation_e interpolation,
               mpl::bool_ resample_,  // Avoid name clash with resample() function
               float alpha,
               mpl::bool_ norm,
               float radius)
{
    // Validate input_array
    nb::ndarray<nb::device::cpu, nb::ro, nb::c_contig> input_array;
    if (!nb::try_cast(input_array_obj, input_array)) {
        // Unknown casting error
        throw std::invalid_argument("Could not cast input array to a nanobind ndarray");
    }

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

    // Validate output array
    nb::ndarray<nb::device::cpu, nb::c_contig> output_array;
    if (!nb::try_cast(output_array_obj, output_array, false)) {
        // nb::ndarray has no flags(). Try casting again without C-contiguous flag
        nb::ndarray<nb::device::cpu> output_array_any_contig;
        if (nb::try_cast(output_array_obj, output_array_any_contig, false)) {
            throw std::invalid_argument("Output array must be C-contiguous");
        }

        // nb::ndarray has no is_writable(). Try casting again as read-only
        nb::ndarray<nb::device::cpu, nb::ro> output_array_ro;
        if (nb::try_cast(output_array_obj, output_array_ro, false)) {
            throw std::invalid_argument("Output array must be writeable");
        }

        // Unknown casting error
        throw std::invalid_argument("Could not cast output array to a nanobind ndarray");
    }

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

    if (output_array.dtype() != dtype) {
        throw std::invalid_argument("Input and output arrays have mismatched types");
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
        // Raises Python AttributeError if no such attribute
        bool is_affine = mpl::is_true(transform.attr("is_affine"));

        if (is_affine) {
            mpl::convert(params.affine, transform, "transform");
            params.is_affine = is_affine;
        } else {
            transform_mesh = _get_transform_mesh(
                transform, output_array.shape(0), output_array.shape(1));
            params.transform_mesh = transform_mesh->data();
            params.is_affine = false;
        }
    }

    if (auto resampler =
            (ndim == 2) ? (
                (dtype == nb::dtype<std::uint8_t>()) ? resample<agg::gray8> :
                (dtype == nb::dtype<std::int8_t>()) ? resample<agg::gray8> :
                (dtype == nb::dtype<std::uint16_t>()) ? resample<agg::gray16> :
                (dtype == nb::dtype<std::int16_t>()) ? resample<agg::gray16> :
                (dtype == nb::dtype<float>()) ? resample<agg::gray32> :
                (dtype == nb::dtype<double>()) ? resample<agg::gray64> :
                nullptr) : (
            // ndim == 3
                (dtype == nb::dtype<std::uint8_t>()) ? resample<agg::rgba8> :
                (dtype == nb::dtype<std::int8_t>()) ? resample<agg::rgba8> :
                (dtype == nb::dtype<std::uint16_t>()) ? resample<agg::rgba16> :
                (dtype == nb::dtype<std::int16_t>()) ? resample<agg::rgba16> :
                (dtype == nb::dtype<float>()) ? resample<agg::rgba32> :
                (dtype == nb::dtype<double>()) ? resample<agg::rgba64> :
                nullptr)) {
        nb::gil_scoped_release release;
        resampler(
            input_array.data(), input_array.shape(1), input_array.shape(0),
            output_array.data(), output_array.shape(1), output_array.shape(0),
            params);
    } else {
        throw std::invalid_argument("arrays must be of dtype byte, short, float32 or float64");
    }
}

[[noreturn]] static void
raise_image_comparison_failure(std::string msg)
{
    auto exceptions = nb::module_::import_("matplotlib.testing.exceptions");
    auto ImageComparisonFailure = exceptions.attr("ImageComparisonFailure");
    PyErr_SetString(ImageComparisonFailure.ptr(), msg.c_str());
    throw nb::python_error();
}


// This is used by matplotlib.testing.compare to calculate RMS and a difference image.
static nb::tuple
calculate_rms_and_diff(const mpl::defer<InputImageArray> &expected_image_obj,
                       const mpl::defer<InputImageArray> &actual_image_obj)
{
    std::optional<InputImageArray> expected_image_opt;
    std::optional<InputImageArray> actual_image_opt;

    try {
        expected_image_opt = expected_image_obj.value("expected_image");
        actual_image_opt = actual_image_obj.value("actual_image");
    } catch (const std::exception &e) {
        raise_image_comparison_failure(e.what());
    }

    auto expected_image = expected_image_opt.value();
    auto actual_image = actual_image_opt.value();

    auto height = expected_image.shape(0);
    auto width = expected_image.shape(1);
    auto depth = expected_image.shape(2);

    if (depth != 3 && depth != 4) {
        raise_image_comparison_failure(mpl::format(
            "expected_image must be RGB or RGBA but has depth {}", depth));
    }

    if (!expected_image.is_same_shape(actual_image)) {
        raise_image_comparison_failure(mpl::format(
            "Image shapes do not match. expected_image is {}, actual_image is {}",
            expected_image.shape_str(), actual_image.shape_str()));
    }

    auto diff_image = OutputImageArray(height, width);

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

    return nb::make_tuple(sqrt(total), diff_image);
}


NB_MODULE(_image, m)
{
    nb::enum_<interpolation_e>(m, "_InterpolationType")
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
