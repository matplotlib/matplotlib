/* -*- mode: c++; c-basic-offset: 4 -*- */

#ifndef __PY_EXCEPTIONS_H__
#define __PY_EXCEPTIONS_H__

#include <exception>
#include <stdexcept>

namespace py
{
class exception : public std::exception
{
  public:
    const char *what() const throw()
    {
        return "python error has been set";
    }
};
}

#define CALL_CPP_FULL(name, a, cleanup, errorcode)                                                 \
    try                                                                                            \
    {                                                                                              \
        a;                                                                                         \
    }                                                                                              \
    catch (const py::exception &e)                                                                 \
    {                                                                                              \
        {                                                                                          \
            cleanup;                                                                               \
        }                                                                                          \
        return (errorcode);                                                                        \
    }                                                                                              \
    catch (const std::bad_alloc)                                                                   \
    {                                                                                              \
        PyErr_Format(PyExc_MemoryError, "In %s: Out of memory", (name));                         \
        {                                                                                          \
            cleanup;                                                                               \
        }                                                                                          \
        return (errorcode);                                                                        \
    }                                                                                              \
    catch (const std::overflow_error &e)                                                           \
    {                                                                                              \
        PyErr_Format(PyExc_OverflowError, "In %s: %s", (name), e.what());                        \
        {                                                                                          \
            cleanup;                                                                               \
        }                                                                                          \
        return (errorcode);                                                                        \
    }                                                                                              \
    catch (char const *e)                                                                          \
    {                                                                                              \
        PyErr_Format(PyExc_RuntimeError, "In %s: %s", (name), e);                                \
        {                                                                                          \
            cleanup;                                                                               \
        }                                                                                          \
        return (errorcode);                                                                        \
    }                                                                                              \
    catch (...)                                                                                    \
    {                                                                                              \
        PyErr_Format(PyExc_RuntimeError, "Unknown exception in %s", (name));                       \
        {                                                                                          \
            cleanup;                                                                               \
        }                                                                                          \
        return (errorcode);                                                                        \
    }

#define CALL_CPP_CLEANUP(name, a, cleanup) CALL_CPP_FULL(name, a, cleanup, NULL)

#define CALL_CPP(name, a) CALL_CPP_FULL(name, a, , NULL)

#define CALL_CPP_INIT(name, a) CALL_CPP_FULL(name, a, , -1)

#endif
