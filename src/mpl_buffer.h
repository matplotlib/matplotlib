/* -*- mode: c++; c-basic-offset: 4 -*- */

#ifndef MPL_BUFFER_H
#define MPL_BUFFER_H

#define PY_SSIZE_T_CLEAN
#include <nanobind/nanobind.h>

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "mpl_utils.h"
#include "mpl_shape.h"

namespace mpl {

namespace nb = nanobind;
using namespace nanobind::literals;

namespace internal {

template <typename T>
constexpr const char *get_buffer_format()
{
    using U = std::remove_cv_t<T>;

    if constexpr (std::is_same_v<U, bool>) {
        return "?";
    } else if constexpr (std::is_same_v<U, char>) {
        return "c";
    } else if constexpr (std::is_integral_v<U> && sizeof(U) == 1) {
        return std::is_signed_v<U> ? "b" : "B";
    } else {
        return nullptr;
    }
}

} // namespace internal


template <typename Scalar, typename Shape>
class buffer_info
{
public:
    using ShapeTraits = internal::shape_traits<Shape>;
    using ShapeType = Shape;

    static_assert(internal::get_buffer_format<Scalar>() != nullptr,
        "mpl::buffer_info: Only single-byte types are currently supported.");

    template <typename... Args, internal::enable_if_dims<ShapeTraits, Args...> = 0>
    explicit buffer_info(Scalar *data, Args... args)
    {
        internal::fill_shape(m_shape, ShapeTraits::base_shape, Dim, args...);

        // If empty due to atleast_3d, etc.; use base_shape with all -1's replaced by 0's
        if (internal::size(m_shape, Dim) == 0) {
            for (size_t i = 0; i < Dim; ++i) {
                m_shape[i] = ShapeTraits::base_shape[i] == -1 ? 0 : ShapeTraits::base_shape[i];
            }
        }

        if (!data && internal::size(m_shape, Dim) > 0) {
            throw std::invalid_argument(mpl::format(
                "mpl::buffer_info: data cannot be null for shape {}",
                internal::shape_str(m_shape, Dim)));
        }

        // Computes the C-contiguous strides from the provided shape
        m_strides[Dim - 1] = 1;
        for (size_t i = Dim - 1; i != 0; --i) {
            m_strides[i - 1] = m_strides[i] * m_shape[i];
        }

        m_data = data;
    }

    constexpr bool is_readonly() { return std::is_const_v<Scalar>; }
    size_t size() const { return internal::size(m_shape, Dim); }
    size_t ndim() const { return Dim; }
    size_t shape(size_t i) const { return static_cast<size_t>(m_shape[i]); }
    size_t nbytes() const { return sizeof(Scalar) * size(); }
    Scalar *data() const { return m_data; }
    void set_owner(nb::object owner) { m_owner = owner; }
    nb::object get_owner() const { return m_owner; }

    void fill_view(Py_buffer *view, int flags)
    {
        auto needs_shape = ((flags & PyBUF_ND) == PyBUF_ND);
        auto needs_strides = ((flags & PyBUF_STRIDES) == PyBUF_STRIDES);
        auto format = internal::get_buffer_format<Scalar>();

        view->buf = const_cast<std::remove_cv_t<Scalar> *>(m_data);
        view->itemsize = 1;
        view->len = nbytes();
        view->readonly = is_readonly() ? 1 : 0;
        view->ndim = ndim();
        view->format = (flags & PyBUF_FORMAT) ? const_cast<char *>(format) : nullptr;
        view->shape = needs_shape ? m_shape : nullptr;
        view->strides = needs_strides ? m_strides : nullptr;
        view->suboffsets = nullptr;
        view->internal = this;
    }

private:
    static constexpr size_t Dim = ShapeTraits::ndim;

    // Optional Python object which owns m_data. If not set, the
    // exporting object is responsible for keeping m_data alive.
    nb::object m_owner {};

    Scalar *m_data = nullptr;
    nb::ssize_t m_shape[Dim] {};
    nb::ssize_t m_strides[Dim] {};
};


template <typename T>
inline int getbuffer(PyObject *obj, Py_buffer *view, int flags)
{
    using BufferInfo = decltype(std::declval<T>().create_buffer_info());

    if (view == nullptr) {
        PyErr_SetString(PyExc_BufferError, "getbuffer: view is NULL");
        return -1;
    }

    view->obj = nullptr;

    T *self = nullptr;
    if (!nb::try_cast(nb::handle(obj), self)) {
        PyErr_SetString(PyExc_BufferError, "getbuffer: obj is the wrong type");
        return -1;
    }

    if ((flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS) {
        PyErr_SetString(PyExc_BufferError, "getbuffer: Only C-contiguity is supported");
        return -1;
    }

    BufferInfo info = nullptr;

    try {
        info = self->create_buffer_info();
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_BufferError, e.what());
        return -1;
    }

    if (!info) {
        PyErr_NoMemory();
        return -1;
    }

    if (info->is_readonly() && (flags & PyBUF_WRITABLE) == PyBUF_WRITABLE) {
        delete info;
        PyErr_SetString(PyExc_BufferError, "getbuffer: object is not writable");
        return -1;
    }

    Py_INCREF(obj);
    view->obj = obj;
    info->fill_view(view, flags);

    return 0;
}


template <typename T>
inline void releasebuffer(PyObject *, Py_buffer *view)
{
    using BufferInfo = decltype(std::declval<T>().create_buffer_info());

    BufferInfo info = static_cast<BufferInfo>(view->internal);

    delete info;
    view->internal = nullptr;
}


template <typename T>
static PyType_Slot buffer_slots[] = {
    {Py_bf_getbuffer, (void *)getbuffer<T>},
    {Py_bf_releasebuffer, (void *)releasebuffer<T>},
    {0, nullptr}
};

} // namespace mpl

#endif // MPL_BUFFER_H
