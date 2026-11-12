/* -*- mode: c++; c-basic-offset: 4 -*- */

#ifndef MPL_ARRAY_H
#define MPL_ARRAY_H

#define PY_SSIZE_T_CLEAN
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "mpl_utils.h"
#include "mpl_shape.h"

namespace mpl {

namespace internal {

template <typename T>
constexpr const char *numpy_dtype()
{
    using U = std::remove_cv_t<T>;
    constexpr bool is_sint = std::is_integral_v<U> && std::is_signed_v<U>;
    constexpr bool is_uint = std::is_integral_v<U> && !std::is_signed_v<U>;

    if constexpr (std::is_same_v<U, bool>) { return "b1"; }
    else if constexpr (is_sint && sizeof(U) == 1) { return "i1"; }
    else if constexpr (is_sint && sizeof(U) == 2) { return "i2"; }
    else if constexpr (is_sint && sizeof(U) == 4) { return "i4"; }
    else if constexpr (is_sint && sizeof(U) == 8) { return "i8"; }
    else if constexpr (is_uint && sizeof(U) == 1) { return "u1"; }
    else if constexpr (is_uint && sizeof(U) == 2) { return "u2"; }
    else if constexpr (is_uint && sizeof(U) == 4) { return "u4"; }
    else if constexpr (is_uint && sizeof(U) == 8) { return "u8"; }
    else if constexpr (std::is_floating_point_v<U> && sizeof(U) == 4) { return "f4"; }
    else if constexpr (std::is_floating_point_v<U> && sizeof(U) == 8) { return "f8"; }

    return nullptr;
}

} // namespace internal


// This is heavily based on ndarray from nanobind
template <typename Scalar, typename Shape>
class array
{
public:
    using ShapeTraits = internal::shape_traits<Shape>;
    using NdArrayType = nb::ndarray<Scalar, Shape, nb::numpy, nb::device::cpu, nb::c_contig>;
    using ShapeType = Shape;

    static array unmanaged_empty()
    {
        static_assert(ShapeTraits::num_free_dim >= 1,
            "unmanaged_empty() requires at least one free dimension");

        return unmanaged_empty_impl(std::make_index_sequence<ShapeTraits::num_free_dim>());
    }

    template <typename... Args, internal::enable_if_dims<ShapeTraits, Args...> = 0>
    static array unmanaged(Scalar *data, Args... args)
    {
        return array(Unmanaged{}, data, args...);
    }

    template <typename... Args, internal::enable_if_dims<ShapeTraits, Args...> = 0>
    explicit array(Args... args)
    {
        size_t shape[Dim];
        internal::fill_shape(shape, ShapeTraits::base_shape, Dim, args...);

        size_t nbytes = sizeof(Scalar) * internal::size(shape, Dim);
        unsigned char *data = nullptr;
        nb::capsule owner;

        if (nbytes > 0) {
            data = new unsigned char[nbytes];
            owner = nb::capsule(data, [](void *p) noexcept {
                delete[] static_cast<unsigned char *>(p);
            });
        }

        auto ndarray = NdArrayType(data, Dim, shape, owner);
        init(&ndarray);
    }

    explicit array(const nb::object in_object, std::string arg_name = "Array")
    {
        NdArrayType ndarray;
        nb::object object = in_object;

        // Attempt 1 - Use the input object. On fail, convert to numpy array.
        if (!nb::try_cast(object, ndarray)) {
            object = asarray(object, internal::numpy_dtype<Scalar>());
        }

        // Attempt 2 - Use the numpy array. On fail, reshape and try one last time.
        if (!nb::try_cast(object, ndarray)) {
            auto numpy_arr_ndim = nb::cast<ssize_t>(object.attr("ndim"));
            auto numpy_arr_size = nb::cast<ssize_t>(object.attr("size"));

            // Only reshape if dimensions are equal
            if (numpy_arr_ndim == Dim && numpy_arr_size == 0) {
                nb::tuple_builder tuple_builder(Dim);

                // FIXME: This could be compile-time
                for (size_t i = 0; i < Dim; ++i) {
                    tuple_builder.put(ShapeTraits::base_shape[i] == -1 ? 0 : ShapeTraits::base_shape[i]);
                }

                object = object.attr("reshape")(tuple_builder.commit());
                nb::try_cast(object, ndarray);
            }
        }

        if (!ndarray.is_valid()) {
            nb::ndarray<Scalar> generic_ndarray;

            if (nb::try_cast(object, generic_ndarray)) {
                internal::check_shape(arg_name.c_str(), Dim, ShapeTraits::base_shape,
                    static_cast<size_t>(generic_ndarray.ndim()),
                    generic_ndarray.shape_ptr());
            }

            throw std::invalid_argument(mpl::format(
                "{} could not be converted to a nanobind ndarray", arg_name));
        }

        init(&ndarray);
    }

    template <typename OtherType>
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

    template <typename Container>
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

    NdArrayType ndarray() const
    {
        if (!m_ndarray) {
            throw std::runtime_error(
                "Cannot return a nanobind ndarray from an unmanaged mpl::array");
        }

        return m_ndarray.value();
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

    template <typename... Args, internal::enable_if_dims<ShapeTraits, Args...> = 0>
    explicit array(Unmanaged, Scalar *data, Args... args)
    {
        int64_t shape[Dim];
        internal::fill_shape(shape, ShapeTraits::base_shape, Dim, args...);

        init(nullptr, shape, data);
    }

    void init(NdArrayType *ndarray, int64_t *shape = nullptr, Scalar *data = nullptr)
    {
        if (!ndarray) {
            if (!shape) {
                throw std::invalid_argument(
                    "mpl::array::init(): shape cannot be null if ndarray is null");
            }
            if (!data && internal::size(shape, Dim) > 0) {
                throw std::invalid_argument(mpl::format(
                    "mpl::array::init(): data cannot be null for shape {}",
                    internal::shape_str(shape, Dim)));
            }
        }

        for (size_t i = 0; i < Dim; ++i) {
            m_shape[i] = ndarray ? ndarray->shape(static_cast<nb::ssize_t>(i)) : shape[i];
        }

        // If empty due to atleast_3d, etc.; use base_shape with all -1's replaced by 0's
        if (internal::size(m_shape, Dim) == 0) {
            for (size_t i = 0; i < Dim; ++i) {
                m_shape[i] = ShapeTraits::base_shape[i] == -1 ? 0 : ShapeTraits::base_shape[i];
            }
        }

        // Computes the C-contiguous strides from the provided shape
        m_strides[Dim - 1] = 1;
        for (size_t i = Dim - 1; i != 0; --i) {
            m_strides[i - 1] = m_strides[i] * m_shape[i];
        }

        if (ndarray) {
            m_ndarray = *ndarray;
            m_data = ndarray->data();
        } else {
            m_data = data;
        }
    }

    static constexpr size_t Dim = ShapeTraits::ndim;

    std::optional<NdArrayType> m_ndarray;
    Scalar *m_data = nullptr;
    int64_t m_shape[Dim] { };
    int64_t m_strides[Dim] { };
};

template <typename Scalar, typename Shape>
struct defer<array<Scalar, Shape>> : public defer_base<array<Scalar, Shape>>
{
    using defer_base<array<Scalar, Shape>>::defer_base;
    static constexpr auto Name = const_name("numpy.ndarray");

    void assign(array<Scalar, Shape> &dst, std::string arg_name) const
    {
        dst = array<Scalar, Shape>(*this, arg_name);
    }

    array<Scalar, Shape> value(std::string arg_name) const
    {
        return array<Scalar, Shape>(*this, arg_name);
    }
};

} // namespace mpl


namespace nanobind::detail {

template <typename Scalar, typename Shape>
struct type_caster<mpl::array<Scalar, Shape>> {
public:
    static constexpr auto Name = const_name("numpy.ndarray");

    bool from_python(handle, uint8_t, cleanup_list *) noexcept {
        static_assert(mpl::internal::always_false_v<Scalar>,
            "Wrap mpl::array arguments with mpl::defer()");
        return false;
    }

    static handle from_cpp(const mpl::array<Scalar, Shape> &src,
        rv_policy policy, cleanup_list *cleanup) noexcept
    {
        using NdArrayType = typename mpl::array<Scalar, Shape>::NdArrayType;
        return type_caster<NdArrayType>::from_cpp(src.ndarray(), policy, cleanup);
    }
};

} // namespace nanobind::detail

#endif // MPL_ARRAY_H
