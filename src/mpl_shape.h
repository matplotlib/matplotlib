/* -*- mode: c++; c-basic-offset: 4 -*- */

#ifndef MPL_SHAPE_H
#define MPL_SHAPE_H

#define PY_SSIZE_T_CLEAN
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "mpl_utils.h"

namespace mpl {

using ssize_t = Py_ssize_t;

namespace nb = nanobind;
using namespace nb::literals;

using nb::shape;
using nb::ndim;

namespace internal {

template <typename S>
struct shape_traits;

template <ssize_t... Is>
struct shape_traits<mpl::shape<Is...>> {
    static_assert(sizeof...(Is) > 0, "Shape must have at least one dimension");

    static constexpr size_t ndim = sizeof...(Is);
    static constexpr size_t num_free_dim = ((Is == -1 ? 1 : 0) + ...);
    static constexpr ssize_t base_shape[sizeof...(Is)] = { Is... };
};


// Only enables when there is exactly one integral argument per -1 in the shape
template <typename ShapeTraits, typename... Args>
using enable_if_dims = std::enable_if_t<
    (sizeof...(Args) == ShapeTraits::num_free_dim) &&
    (std::is_integral_v<Args> && ...), int>;


// Checks that value is >= 0 and <= numeric_limits<ssize_t>::max()
template <typename T>
bool is_valid_for_shape(T value)
{
    static_assert(std::is_integral_v<T>);

    if constexpr (std::is_signed_v<T>) {
        if (value < 0) { return false; }
    }

    constexpr auto max_value = std::numeric_limits<ssize_t>::max();
    return static_cast<uint64_t>(value) <= static_cast<uint64_t>(max_value);
}


// Returns the product of all elements of shape and ensures they are in range
template <typename ShapeType>
inline size_t size(ShapeType *shape, size_t ndim)
{
    if (!ndim) return 0;

    size_t result = 1;
    for (size_t i = 0; i < ndim; ++i) {
        if (!is_valid_for_shape(shape[i])) {
            throw std::out_of_range(
                "size(): values in shape must be >= 0 and <= ssize_t max");
        }

        result *= shape[i];
    }

    return result;
}


// Replaces all -1 values in a base shape with the provided arguments.
// For example, a base shape of mpl::shape<-1, 2, -1, 4, -1> and
// fill_shape(..., 1, 3, 5) would result in { 1, 2, 3, 4, 5 }
template <typename ShapeType, typename... Args>
inline void fill_shape(
    ShapeType *out_shape, const ssize_t *base_shape, size_t ndim, Args... in_args)
{
    static_assert(
        std::numeric_limits<ShapeType>::max() >= std::numeric_limits<ssize_t>::max(),
        "out_shape must be able to hold a ssize_t max");

    if (!(is_valid_for_shape(in_args) && ...)) {
        throw std::out_of_range("in_args value must be >= 0 and <= ssize_t max");
    }

    const ShapeType args[] = { (ShapeType) in_args..., 0 };
    size_t a = 0;

    for (size_t i = 0; i < ndim; ++i) {
        out_shape[i] = static_cast<ShapeType>(
            base_shape[i] == -1 ? args[a++] : base_shape[i]);
    }
}


// Makes a shape string "(N, 2, 3)". Used by check_shape()
template <typename ShapeType>
inline std::string shape_str(ShapeType shape, size_t ndim)
{
    std::string s = "(";

    for (size_t i = 0; i < ndim; ++i) {
        s += (i > 0 ? ", " : "") + (shape[i] == -1 ? "N" : std::to_string(shape[i]));
    }

    s += ")";

    return s;
}


// Throws if expected shape does not equal actual shape.
// -1 values in expected_shape act as a wildcard
inline void check_shape(const char *name,
    size_t expected_ndim, const ssize_t *expected_shape,
    size_t actual_ndim, const int64_t *actual_shape)
{
    if (expected_ndim != actual_ndim) {
        throw std::invalid_argument(mpl::format(
            "{} should have {} dimension{}, got {}",
            name, expected_ndim, expected_ndim == 1 ? "" : "s", actual_ndim));
    }

    // Sometimes things come through as atleast_3d, etc., but they're empty, so
    // don't bother enforcing the shape.
    for (size_t i = 0; i < actual_ndim; ++i) {
        if (actual_shape[i] == 0) {
            return;
        }
    }

    for (size_t i = 0; i < expected_ndim; ++i) {
        if (expected_shape[i] != -1 && (expected_shape[i] != actual_shape[i])) {
            throw std::invalid_argument(mpl::format(
                "{} must have shape {}, got {}", name,
                shape_str(expected_shape, expected_ndim),
                shape_str(actual_shape, actual_ndim)));
        }
    }
}


} // namespace internal

} // namespace mpl

#endif // MPL_SHAPE_H
