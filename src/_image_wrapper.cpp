#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>

namespace nb = nanobind;
using namespace nanobind::literals;

#include "mplutils.h"
#include "_image_resample.h"
#include "nb_converters.h"


using TransformMeshArray = nb::ndarray<double, nb::ndim<2>, nb::numpy, nb::c_contig>;
using DiffArray = nb::ndarray<unsigned char, nb::ndim<3>, nb::numpy, nb::c_contig>;


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

static TransformMeshArray
_get_transform_mesh(const nb::object& transform, size_t width, size_t height)
{
    /* TODO: Could we get away with float, rather than double, arrays here? */

    /* Given a non-affine transform object, create a mesh that maps
    every pixel center in the output image to the input image.  This is used
    as a lookup table during the actual resampling. */

    // If attribute doesn't exist, raises Python AttributeError
    auto inverse = transform.attr("inverted")();

    size_t mesh_dims[2] = {width*height, 2};
    auto input_mesh = mpl_make_numpy_array<TransformMeshArray>({mesh_dims[0], mesh_dims[1]});
    double *p = input_mesh.data();

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            // The convention for the supplied transform is that pixel centers
            // are at 0.5, 1.5, 2.5, etc.
            *p++ = (double)x + 0.5;
            *p++ = (double)y + 0.5;
        }
    }

    nb::object output_mesh = inverse.attr("transform")(input_mesh);

    TransformMeshArray output_mesh_array;
    if (!nb::try_cast(output_mesh, output_mesh_array)) {
        throw std::runtime_error(
            "Inverse transformed mesh could not be converted to a double array");
    }

    if (output_mesh_array.ndim() != 2) {
        throw std::runtime_error(nb::str(
            "Inverse transformed mesh array should be 2D not {}D").format(
            output_mesh_array.ndim()).c_str());
    }

    // An undersized mesh would be read out of bounds by the resampler.
    if (output_mesh_array.shape(0) != mesh_dims[0] || output_mesh_array.shape(1) != mesh_dims[1]) {
        throw std::runtime_error(nb::str(
            "Inverse transformed mesh array should have shape ({}, {}) not ({}, {})").format(
                mesh_dims[0], mesh_dims[1],
                output_mesh_array.shape(0), output_mesh_array.shape(1)).c_str());
    }

    return output_mesh_array;
}

static void
_fill_params_affine(const nb::object& transform, agg::trans_affine& affine)
{
    nb::object array_object = transform.attr("__array__")();
    nb::ndarray<double, nb::c_contig> array;

    if (!nb::try_cast(array_object, array)) {
        throw std::invalid_argument("Could not convert affine transformation matrix");
    }

    if (array.ndim() != 2 || array.shape(0) != 3 || array.shape(1) != 3) {
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

// Use generic nb::ndarrays without a dtype for input and output arrays
static void
image_resample(nb::ndarray<nb::numpy, nb::c_contig> &input_array,
               nb::ndarray<nb::numpy, nb::c_contig> &output_array,
               const nb::object& transform,
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
        throw std::invalid_argument(nb::str(
            "3D input array must be RGBA with shape (M, N, 4), has trailing dimension of {}").format(
                input_array.shape(2)).c_str());
    }

    // Validate output array
    auto out_ndim = output_array.ndim();

    if (out_ndim != ndim) {
        throw std::invalid_argument(
            "Input ({}D) and output ({}D) arrays have different dimensionalities"_s.format(
                ndim, out_ndim).c_str());
    }

    if (out_ndim == 3 && output_array.shape(2) != 4) {
        throw std::invalid_argument(nb::str(
            "3D output array must be RGBA with shape (M, N, 4), "
            "has trailing dimension of {}").format(output_array.shape(2)).c_str());
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
    TransformMeshArray transform_mesh;

    // Validate transform
    if (transform.is_none()) {
        params.is_affine = true;
    } else {
        // Raises Python AttributeError if no such attribute or TypeError if cast fails
        bool is_affine = nb::cast<bool>(transform.attr("is_affine"));

        if (is_affine) {
            _fill_params_affine(transform, params.affine);
            params.is_affine = true;
        } else {
            transform_mesh = _get_transform_mesh(
                transform, output_array.shape(1), output_array.shape(0));
            params.transform_mesh = transform_mesh.data();
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
raise_image_comparison_failure(const char *msg)
{
    auto exceptions = nb::module_::import_("matplotlib.testing.exceptions");
    auto ImageComparisonFailure = exceptions.attr("ImageComparisonFailure");
    PyErr_SetString(ImageComparisonFailure.ptr(), msg);
    throw nb::python_error();
}

// This is used by matplotlib.testing.compare to calculate RMS and a difference image.
static nb::tuple
calculate_rms_and_diff(nb::ndarray<const uint8_t, nb::c_contig> &expected_image,
                       nb::ndarray<const uint8_t, nb::c_contig> &actual_image)
{


    for (const auto & [image, name] : {std::pair{expected_image, "Expected"},
                                       std::pair{actual_image, "Actual"}})
    {
        if (image.ndim() != 3) {
            raise_image_comparison_failure(nb::str(
                "{} image must be 3-dimensional, but is {}-dimensional").format(
                    name, image.ndim()).c_str());
        }
    }

    auto height = expected_image.shape(0);
    auto width = expected_image.shape(1);
    auto depth = expected_image.shape(2);

    if (depth != 3 && depth != 4) {
        raise_image_comparison_failure(nb::str(
            "Image must be RGB or RGBA but has depth {}").format(depth).c_str());
    }

    if (height != actual_image.shape(0) || width != actual_image.shape(1) ||
            depth != actual_image.shape(2)) {
        raise_image_comparison_failure(nb::str(
            "Image sizes do not match expected size: {expected_image.shape} "_s
            "actual size {actual_image.shape}").format(
                "expected_image"_a=expected_image, "actual_image"_a=actual_image).c_str());
    }

    auto expected = expected_image.view<unsigned char, nb::ndim<3>>();
    auto actual = actual_image.view<unsigned char, nb::ndim<3>>();

    auto diff_image = mpl_make_numpy_array<DiffArray>({height, width, 3});
    auto diff = diff_image.view();

    double total = 0.0;
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            for (size_t k = 0; k < depth; k++) {
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
        "output_array"_a.noconvert(),
        "transform"_a,
        "interpolation"_a = interpolation_e::NEAREST,
        "resample"_a = false,
        "alpha"_a = 1.0f,
        "norm"_a = false,
        "radius"_a = 1.0f,
        image_resample__doc__);

    m.def("calculate_rms_and_diff", &calculate_rms_and_diff,
          "expected_image"_a, "actual_image"_a);
}
