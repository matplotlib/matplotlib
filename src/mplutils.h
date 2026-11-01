/* -*- mode: c++; c-basic-offset: 4 -*- */

/* Small utilities that are shared by most extension modules. */

#ifndef MPLUTILS_H
#define MPLUTILS_H
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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <array>
#include <cstddef>
#include <string>

namespace py = pybind11;
using namespace pybind11::literals;


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

inline int round_to_int(double v)
{
    return (int)(v + ((v >= 0.0) ? 0.5 : -0.5));
}

inline double round(double v)
{
    return (double)round_to_int(v);
}

template <typename... Args>
inline std::string format(const char *fmt, Args &&...args)
{
    return std::string(py::str(fmt).format(std::forward<Args>(args)...));
}


inline int check_any(PyObject *) { return 1; }

template <typename T>
class defer : public py::object
{
    PYBIND11_OBJECT_DEFAULT(defer, py::object, check_any)

    template <typename... Args>
    T value(Args &&...args) const { return T(*this, args...); }
};


template <typename T, typename... Args>
inline auto asarray(const py::object &object)
{
    return py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(object);
}

inline auto asarray(const py::object &object)
{
    return py::array::ensure(object, py::array::c_style);
}


} // namespace mpl


// Helper for std::visit.
template<typename... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<typename... Ts> overloaded(Ts...) -> overloaded<Ts...>;

#endif
