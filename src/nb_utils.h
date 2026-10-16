/* -*- mode: c++; c-basic-offset: 4 -*- */

#ifndef MPL_NB_UTILS_H
#define MPL_NB_UTILS_H

#define PY_SSIZE_T_CLEAN
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>
#include <string>

namespace nb = nanobind;
using namespace nb::literals;

namespace mpl {

// A nanobind wrapper to safely cast any object to bool using PyObject_IsTrue()
class BoolLike : public nb::object
{
    NB_OBJECT(BoolLike, nb::object, "bool", nb::object::check_)

    operator bool() const
    {
        return PyObject_IsTrue(this->ptr());
    }
};


// A nanobind wrapper to act as a placeholder during Python -> C++ calls.
// An ArrayLike is later converted to an mpl::Array
class ArrayLike : public nb::object {
    NB_OBJECT(ArrayLike, nb::object, "ArrayLike", nb::object::check_)
};


// raise_error() without a nanobind builtin_exception raises a Python RuntimeError
template <typename... Args>
[[noreturn]] void raise_error(const char *fmt = "", Args &&...args) {
    throw std::runtime_error(nb::str(fmt).format(std::forward<Args>(args)...).c_str());
}

// raise_error() with a nanobind builtin_exception (nb::type_error -> TypeError, etc)
template <typename T, typename... Args>
[[noreturn]] void raise_error(T, const char *fmt = "", Args &&...args) {
    throw T(nb::str(fmt).format(std::forward<Args>(args)...).c_str());
}


namespace internal {

// Converts a C++ typename at compile-time into a numpy dtype kind
template <typename T>
[[maybe_unused]] static const char *get_numpy_dtype_kind()
{
    using U = std::remove_cv_t<T>;

    if constexpr (std::is_same_v<U, bool>) {
        return "b";
    } else if constexpr (std::is_floating_point_v<U>) {
        return "f";
    } else if constexpr (std::is_signed_v<U>) {
        return "i";
    } else {
        return "u";
    }
}


// Converts an ArrayLike object into a numpy.ndarray using numpy.asarray()
[[maybe_unused]] static nb::object numpy_asarray(nb::object obj, const char *dtype_kind)
{
    static auto numpy = nb::module_::import_("numpy");
    static auto asarray = numpy.attr("asarray");
    static auto dtype = numpy.attr("dtype");

    return asarray(obj, dtype(dtype_kind), "C");
}


// Makes a shape string "(N, 2, 3)". Used by check_shape()
[[maybe_unused]] static std::string shape_str(size_t ndim, const size_t *shape)
{
    std::string s = "(";

    for (size_t i = 0; i < ndim; i++) {
        s += (i > 0 ? ", " : "") + (shape[i] > 0 ? std::to_string(shape[i]) : "N");
    }

    s += ")";
    return s;
}


// Throws if expected shape does not equal actual shape.
// 0 values in expected_shape act as a wildcard
[[maybe_unused]] static void check_shape(const char *name,
    size_t expected_ndim, const size_t *expected_shape,
    size_t actual_ndim, const size_t *actual_shape)
{
    if (expected_ndim != actual_ndim) {
        raise_error(nb::value_error, "'{}' should be an {}-dimensional array, got {}",
            name, expected_ndim, actual_ndim);
    }

    // Sometimes things come through as atleast_3d, etc., but they're empty, so
    // don't bother enforcing the shape.
    for (size_t i = 0; i < actual_ndim; i++) {
        if (actual_shape[i] == 0) {
            return;
        }
    }

    for (size_t i = 0; i < expected_ndim; i++) {
        if (expected_shape[i] > 0 && (expected_shape[i] != actual_shape[i])) {
            raise_error(nb::value_error, "'{}' must have shape {}, got {}", name,
                shape_str(expected_ndim, expected_shape),
                shape_str(actual_ndim, actual_shape));
        }
    }
}


// See "Returning arrays..." on https://nanobind.readthedocs.io/en/latest/ndarray.html
[[maybe_unused]] static void alloc_capsule(
    void *&data, std::initializer_list<size_t> &shape,
    nb::capsule &owner, size_t scalar_size)
{
    size_t total_size = scalar_size;
    for (size_t n : shape) {
        total_size *= n;
    }

    data = total_size ? new unsigned char[total_size] : nullptr;
    owner = nb::capsule(data, [](void *p) noexcept {
        delete[] static_cast<unsigned char *>(p);
    });
}


} // namespace internal


using ArrayShape = std::initializer_list<size_t>;

template <typename Scalar, size_t Dim>
class ArrayView;

template <typename Scalar, size_t Dim>
class Array
{
public:
    using NdArrayType = nb::ndarray<Scalar, nb::c_contig, nb::numpy, nb::device::cpu>;
    using ViewType = ArrayView<Scalar, Dim>;

    Array(ArrayShape expected_shape, const char *arg_name, ArrayLike arraylike)
    {
        if (!nb::try_cast(arraylike, m_ndarray)) {
            const char *dtype_kind = internal::get_numpy_dtype_kind<Scalar>();

            if (!nb::try_cast(internal::numpy_asarray(arraylike, dtype_kind), m_ndarray)) {
                throw nb::value_error("Conversion of ArrayLike to Array failed");
            }
        }

        internal::check_shape(arg_name, expected_shape.size(), expected_shape.begin(),
                              m_ndarray.ndim(), m_ndarray.shape_ptr());
    }

    Array(ArrayShape shape)
    {
        if (shape.size() != Dim) {
            throw std::invalid_argument("Array() called with incompatible shape");
        }

        void *data;
        nb::capsule owner;
        internal::alloc_capsule(data, shape, owner, sizeof(Scalar));

        m_ndarray = NdArrayType(static_cast<Scalar *>(data), shape, owner);
    }

    inline size_t ndim() const { return m_ndarray.ndim(); }
    inline size_t shape(size_t i) const { return m_ndarray.shape(i); }
    inline Scalar *data() const { return m_ndarray.data(); }
    ViewType view() const { return ViewType(*this); }
    NdArrayType ndarray() const { return m_ndarray; }

private:
    NdArrayType m_ndarray;
};


// This is heavily based on ndarray_view in nanobind
template <typename Scalar, size_t Dim>
class ArrayView
{
public:
    using ArrayType = Array<Scalar, Dim>;

    ArrayView(const Array<Scalar, Dim> &array) :
        m_ndarray(array.ndarray()),
        m_data(array.data())
    {
        for (size_t i = 0; i < Dim; ++i) { m_shape[i] = array.shape(i); }
        fill_strides();
    }

    ArrayView(Scalar *data, ArrayShape shape) :
        m_ndarray(std::nullopt),
        m_data(data)
    {
        std::copy(shape.begin(), shape.end(), m_shape);
        fill_strides();
    }

    template <typename... Args>
    inline Scalar &operator()(Args... indices) const
    {
        static_assert(sizeof...(Args) == Dim,
            "ArrayView::operator(): invalid number of arguments");

        const int64_t indices_i64[] { (int64_t) indices... };
        int64_t offset = 0;
        for (size_t i = 0; i < Dim; ++i) {
            offset += indices_i64[i] * m_strides[i];
        }

        return *(m_data + offset);
    }

    size_t ndim() const { return Dim; }
    size_t shape(size_t i) const { return (size_t) m_shape[i]; }
    Scalar *data() const { return m_data; }

private:
    void fill_strides()
    {
        size_t n = 1;

        for (size_t d = Dim; d > 0; --d) {
            size_t i = d - 1;
            m_strides[i] = n;
            n *= m_shape[i];
        }
    }

    std::optional<typename ArrayType::NdArrayType> m_ndarray;
    Scalar *m_data = nullptr;
    int64_t m_shape[Dim] { };
    int64_t m_strides[Dim] { };
};

} // namespace mpl


namespace nanobind::detail {

template <typename Scalar, size_t Dim>
struct type_caster<mpl::Array<Scalar, Dim>> {
    using ArrayType = mpl::Array<Scalar, Dim>;
    using NdArrayType = typename ArrayType::NdArrayType;

public:
    NB_TYPE_CASTER(ArrayType, const_name("Array<") +
        make_caster<Scalar>::Name + const_name(", ") + const_name(Dim) +
        const_name(">"));

    static handle from_cpp(const ArrayType &src, rv_policy policy, cleanup_list *cleanup) {
        return type_caster<NdArrayType>::from_cpp(src.ndarray(), policy, cleanup);
    }
};

} // namespace nanobind::detail

#endif
