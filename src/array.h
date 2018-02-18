/* -*- mode: c++; c-basic-offset: 4 -*- */

/* Utilities to create scalars and empty arrays that behave like the
   Numpy array wrappers in numpy_cpp.h */

#ifndef _SCALAR_H_
#define _SCALAR_H_

namespace array
{

template <typename T, int ND>
class scalar
{
  public:
    T m_value;

    scalar(const T value) : m_value(value)
    {
    }

    T &operator()(int i, int j = 0, int k = 0)
    {
        return m_value;
    }

    const T &operator()(int i, int j = 0, int k = 0) const
    {
        return m_value;
    }

    int dim(size_t i)
    {
        return 1;
    }

    size_t size()
    {
        return 1;
    }
};

template <typename T>
class empty
{
  public:
    typedef empty<T> sub_t;

    empty()
    {
    }

    T &operator()(int i, int j = 0, int k = 0)
    {
        throw std::runtime_error("Accessed empty array");
    }

    const T &operator()(int i, int j = 0, int k = 0) const
    {
        throw std::runtime_error("Accessed empty array");
    }

    sub_t operator[](int i) const
    {
        return empty<T>();
    }

    int dim(size_t i) const
    {
        return 0;
    }

    size_t size() const
    {
        return 0;
    }
};
}

#endif
