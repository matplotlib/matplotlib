/* -*- mode: c++; c-basic-offset: 4 -*- */

#ifndef MPL_PY_ARRAY_H
#define MPL_PY_ARRAY_H

#define PY_SSIZE_T_CLEAN
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "mplutils.h"

namespace mpl {

using ssize_t = Py_ssize_t;
namespace py = pybind11;
using namespace py::literals;

// shape<> and ndim<> are vendored from nanobind
namespace vendored {

template <ssize_t... Is> struct shape {
    static_assert(
        ((Is >= 0 || Is == -1) && ...),
        "The arguments to mpl::shape must either be positive or equal to -1"
    );
};

template <typename> struct ndim_shape;
template <size_t... S> struct ndim_shape<std::index_sequence<S...>> {
    using type = shape<((void) S, -1)...>;
};

template <size_t N>
using ndim = typename ndim_shape<std::make_index_sequence<N>>::type;

} // namespace vendored

using vendored::shape;
using vendored::ndim;


namespace internal {

template <typename S>
struct shape_traits;

template <ssize_t... Is>
struct shape_traits<mpl::shape<Is...>> {
    static_assert(sizeof...(Is) > 0, "Array shape must have at least one dimension");

    static constexpr size_t ndim = sizeof...(Is);
    static constexpr size_t num_free_dim = ((Is == -1 ? 1 : 0) + ...);
    static constexpr ssize_t base_shape[sizeof...(Is)] = { Is... };
};


// Used for static_assert in type_caster::load
template <typename T>
inline constexpr bool always_false_v = false;


// Only enables when there is exactly one integral argument per -1 in the shape
template <typename Traits, typename... Args>
using enable_if_args_match = std::enable_if_t<
    (sizeof...(Args) == Traits::num_free_dim) &&
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
inline std::string shape_str(const ShapeType *shape, size_t ndim)
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
    size_t actual_ndim, const ssize_t *actual_shape)
{
    if (expected_ndim != actual_ndim) {
        throw py::value_error(mpl::format(
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
            throw py::value_error(mpl::format(
                "{} must have shape {}, got {}", name,
                shape_str(expected_shape, expected_ndim),
                shape_str(actual_shape, actual_ndim)));
        }
    }
}

} // namespace internal


// This is heavily based on ndarray from nanobind
template <typename Scalar, typename Shape>
class array
{
public:
    using Traits = internal::shape_traits<Shape>;
    using PyArrayType = py::array_t<Scalar, py::array::c_style | py::array::forcecast>;
    using ShapeType = Shape;

    static array unmanaged_empty()
    {
        static_assert(Traits::num_free_dim >= 1,
            "unmanaged_empty() requires at least one free dimension");

        return unmanaged_empty_impl(std::make_index_sequence<Traits::num_free_dim>());
    }

    template <typename... Args, internal::enable_if_args_match<Traits, Args...> = 0>
    static array unmanaged(Scalar *data, Args... args)
    {
        return array(Unmanaged{}, data, args...);
    }

    template <typename... Args, internal::enable_if_args_match<Traits, Args...> = 0>
    explicit array(Args... args)
    {
        ssize_t shape[Dim];
        internal::fill_shape(shape, Traits::base_shape, Dim, args...);

        auto pyarray = PyArrayType(std::vector<ssize_t>(shape, shape + Dim));
        init(&pyarray);
    }

    explicit array(const py::object object, const char *arg_name = "Array")
    {
        // Create pyarray from object
        auto pyarray = PyArrayType::ensure(object);

        if (!pyarray) {
            throw py::value_error(mpl::format(
                "{} could not be converted to a py::array", arg_name));
        }

        if constexpr (!std::is_const_v<Scalar>) {
            if (!pyarray.writeable()) {
                throw py::value_error(mpl::format("{} is not writable", arg_name));
            }
        }

        internal::check_shape(arg_name, Dim, Traits::base_shape,
                              static_cast<size_t>(pyarray.ndim()), pyarray.shape());

        init(&pyarray);
    }

    template<typename OtherType>
    bool is_same_shape(const OtherType &other) const
    {
        static_assert(std::is_same_v<Shape, typename OtherType::ShapeType>,
            "Shape types must be equal");

        if (size() == 0 && other.size() == 0) { return true; }

        for (size_t i = 0; i < Dim; ++i) {
            if (shape(i) != other.shape(i)) { return false; }
        }

        return true;
    }

    template<typename Container>
    void assign(const Container &source) const
    {
        std::copy(source.begin(), source.end(), m_data);
    }

    template <typename... Args>
    Scalar *data(Args... indices) const
    {
        static_assert(sizeof...(Args) == (size_t) Dim,
            "mpl::array::data(): invalid number of arguments");

        size_t counter = 0;
        int64_t index = 0;
        ((index += int64_t(indices) * m_strides[counter++]), ...);

        return m_data + index;
    }

    template <typename... Args>
    Scalar &operator()(Args... indices) const
    {
        static_assert(sizeof...(Args) == (size_t) Dim,
                      "mpl::array::operator(): invalid number of arguments");

        return *data(indices...);
    }

    PyArrayType pyarray() const
    {
        if (!m_pyarray) {
            throw std::runtime_error(
                "Cannot return a py::array from an unmanaged mpl::array");
        }

        return m_pyarray.value();
    }

    size_t size() const { return internal::size(m_shape, Dim); }
    size_t ndim() const { return Dim; }
    size_t shape(size_t i) const { return static_cast<size_t>(m_shape[i]); }
    size_t itemsize() const { return sizeof(Scalar); }
    size_t nbytes() const { return sizeof(Scalar) * size(); }
    Scalar *data() const { return m_data; }
    Scalar *begin() const { return m_data; }
    Scalar *end() const { return m_data + size(); }
    bool has_elements() const { return size() != 0; }
    std::string shape_str() const { return internal::shape_str(m_shape, Dim); }

private:
    struct Unmanaged {};

    template <size_t... Is>
    static array unmanaged_empty_impl(std::index_sequence<Is...>)
    {
        return unmanaged(nullptr, ((void) Is, 0)...);
    }

    template <typename... Args, internal::enable_if_args_match<Traits, Args...> = 0>
    explicit array(Unmanaged, Scalar *data, Args... args)
    {
        int64_t shape[Dim];
        internal::fill_shape(shape, Traits::base_shape, Dim, args...);

        init(nullptr, shape, data);
    }

    void init(PyArrayType *pyarray, int64_t *shape = nullptr, Scalar *data = nullptr)
    {
        if (!pyarray) {
            if (!shape) {
                throw std::invalid_argument(
                    "mpl::array::init(): shape cannot be null if pyarray is null");
            }
            if (!data && internal::size(shape, Dim) > 0) {
                throw std::invalid_argument(mpl::format(
                    "mpl::array::init(): data cannot be null for shape {}",
                    internal::shape_str(shape, Dim)));
            }
        }

        for (size_t i = 0; i < Dim; ++i) {
            m_shape[i] = pyarray ? pyarray->shape(static_cast<ssize_t>(i)) : shape[i];
        }

        // If empty due to atleast_3d, etc.; use base_shape with all -1's replaced by 0's
        if (internal::size(m_shape, Dim) == 0) {
            for (size_t i = 0; i < Dim; ++i) {
                m_shape[i] = Traits::base_shape[i] == -1 ? 0 : Traits::base_shape[i];
            }
        }

        // Computes the C-contiguous strides from the provided shape
        m_strides[Dim - 1] = 1;
        for (size_t i = Dim - 1; i != 0; --i) {
            m_strides[i - 1] = m_strides[i] * m_shape[i];
        }

        if (pyarray) {
            m_pyarray = *pyarray;

            if constexpr (std::is_const_v<Scalar>) {
                m_data = pyarray->data();
            } else {
                m_data = pyarray->mutable_data();
            }

        } else {
            m_data = data;
        }
    }

    static constexpr size_t Dim = Traits::ndim;

    std::optional<PyArrayType> m_pyarray;
    Scalar *m_data = nullptr;
    int64_t m_shape[Dim] { };
    int64_t m_strides[Dim] { };
};

} // namespace mpl


namespace pybind11::detail {

template <typename Scalar, typename Shape>
struct type_caster<mpl::array<Scalar, Shape>> {
public:
    static constexpr auto name = const_name("numpy.ndarray");

    bool load(handle, bool) {
        static_assert(mpl::internal::always_false_v<Scalar>,
            "Wrap mpl::array arguments with mpl::defer()");
        return false;
    }

    static handle cast(const mpl::array<Scalar, Shape> &src, return_value_policy, handle) {
        return src.pyarray().release();
    }
};

} // namespace pybind11::detail

#endif // MPL_PY_ARRAY_H
