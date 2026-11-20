/* -*- mode: c++; c-basic-offset: 4 -*- */

/* Small utilities that are shared by most extension modules. */

#ifndef MPL_UTILS_H
#define MPL_UTILS_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#ifdef _POSIX_C_SOURCE
#    undef _POSIX_C_SOURCE
#endif
#ifndef _AIX
#ifdef _XOPEN_SOURCE
#    undef _XOPEN_SOURCE
#endif
#endif

// Prevent multiple conflicting definitions of swab from stdlib.h and unistd.h
#if defined(__sun) || defined(sun)
#if defined(_XPG4)
#undef _XPG4
#endif
#if defined(_XPG3)
#undef _XPG3
#endif
#endif

#include <nanobind/nanobind.h>
#include <array>
#include <cstddef>
#include <string>


// 'kind' codes for paths.
enum {
    STOP = 0,
    MOVETO = 1,
    LINETO = 2,
    CURVE3 = 3,
    CURVE4 = 4,
    CLOSEPOLY = 0x4f
};


namespace mpl {

namespace nb = nanobind;
using namespace nb::literals;
using nb::detail::const_name;

namespace internal {

template <typename T> inline constexpr bool always_false_v = false;
inline bool check_any(PyObject *) noexcept { return true; }

struct formatted_string : std::string
{
    formatted_string(std::string s) : std::string(std::move(s)) {}
    operator const char *() const { return c_str(); }
};

} // namespace internal


inline int round_to_int(double v)
{
    return (int)(v + ((v >= 0.0) ? 0.5 : -0.5));
}

inline double round(double v)
{
    return (double)round_to_int(v);
}

template <typename... Args>
inline auto format(const char *fmt, Args &&...args)
{
    return internal::formatted_string(
        nb::str(fmt).format(std::forward<Args>(args)...).c_str());
}

template <typename T>
inline void cast(T &dst, const nb::object &src, std::string arg_name)
{
    if (!nb::try_cast(src, dst)) {
        throw nb::value_error(mpl::format(
            "Could not cast {} to specified type", arg_name));
    }
}

template <typename T>
inline void convert(T &dst, const nb::object src, std::string arg_name)
{
    static_assert(internal::always_false_v<T>,
        "No convert() specialization registered for this type");
}

template <typename T>
struct defer_type_name
{
    static constexpr auto Name = const_name("defer");
};

#define MPL_TYPE_NAME(TYPE, NAME) \
    template<> struct defer_type_name<TYPE> \
        { static constexpr auto Name = const_name(NAME); };

template <typename T>
struct defer_base : public nb::object
{
    NB_OBJECT_DEFAULT(defer_base, object, "defer", internal::check_any)
};

template <typename T>
struct defer : public defer_base<T>
{
    static constexpr auto Name = defer_type_name<T>::Name;

    using defer_base<T>::defer_base;

    T value(std::string arg_name) const
    {
        auto obj = T();
        convert(obj, *this, arg_name);
        return obj;
    }

    void assign(T &dst, std::string arg_name) const
    {
        convert(dst, *this, arg_name);
    }
};

// is_true() and the mpl::bool_ wrapper work around nanobind not handling NumPy bools
inline bool is_true(const nb::object &obj)
{
    return PyObject_IsTrue(obj.ptr()) ? true : false;
}

// cast() overload to support more-lenient bools
inline void cast(bool &dst, const nb::object &src, std::string arg_name)
{
    dst = is_true(src);
}

class bool_ : public nb::object
{
    NB_OBJECT_DEFAULT(bool_, object, "bool", internal::check_any)

    operator bool() const noexcept
    {
        return is_true(*this);
    }
};

// Converts an ArrayLike object into a numpy.ndarray using numpy.asarray()
inline nb::object asarray(const nb::object &obj, const char *dtype = nullptr)
{
    auto numpy = nb::module_::import_("numpy");
    auto asarray = numpy.attr("asarray");

    if (dtype) {
        return asarray(obj, nb::str(dtype), "C");
    } else {
        nb::object arr = asarray(obj, nb::none(), "C");
        nb::object dtype_obj = arr.attr("dtype");

        if (!nb::cast<bool>(dtype_obj.attr("isnative"))) {
            arr = arr.attr("astype")(dtype_obj.attr("newbyteorder")("="), "C");
        }

        return arr;
    }
}

} // namespace mpl


// Helper for std::visit.
template<typename... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<typename... Ts> overloaded(Ts...) -> overloaded<Ts...>;

#endif // MPL_UTILS_H
