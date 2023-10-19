//----------------------------------------------------------------------------
// Anti-Grain Geometry - Version 2.4
// Copyright (C) 2002-2005 Maxim Shemanarev (http://www.antigrain.com)
//
// Permission to copy, use, modify, sell and distribute this software 
// is granted provided this copyright notice appears in all copies. 
// This software is provided "as is" without express or implied
// warranty, and with no claim as to its suitability for any purpose.
//
//----------------------------------------------------------------------------
// Contact: mcseem@antigrain.com
//          mcseemagg@yahoo.com
//          http://www.antigrain.com
//----------------------------------------------------------------------------
#ifndef AGG_ARRAY_INCLUDED
#define AGG_ARRAY_INCLUDED

#include <stddef.h>
#include <string.h>
#include "agg_basics.h"

namespace agg
{

    //-------------------------------------------------------pod_array_adaptor
    template<class T> class pod_array_adaptor
    {
    public:
        typedef T value_type;
        pod_array_adaptor(T* array, unsigned size) : 
            m_array(array), m_size(size) {}

        unsigned size() const { return m_size; }
        const T& operator [] (unsigned i) const { return m_array[i]; }
              T& operator [] (unsigned i)       { return m_array[i]; }
        const T& at(unsigned i) const           { return m_array[i]; }
              T& at(unsigned i)                 { return m_array[i]; }
        T  value_at(unsigned i) const           { return m_array[i]; }

    private:
        T*       m_array;
        unsigned m_size;
    };


    //---------------------------------------------------------pod_auto_array
    template<class T, unsigned Size> class pod_auto_array
    {
    public:
        typedef T value_type;
        typedef pod_auto_array<T, Size> self_type;

        pod_auto_array() {}
        explicit pod_auto_array(const T* c)
        {
            memcpy(m_array, c, sizeof(T) * Size);
        }

        const self_type& operator = (const T* c)
        {
            memcpy(m_array, c, sizeof(T) * Size);
            return *this;
        }

        static unsigned size() { return Size; }
        const T& operator [] (unsigned i) const { return m_array[i]; }
              T& operator [] (unsigned i)       { return m_array[i]; }
        const T& at(unsigned i) const           { return m_array[i]; }
              T& at(unsigned i)                 { return m_array[i]; }
        T  value_at(unsigned i) const           { return m_array[i]; }

    private:
        T m_array[Size];
    };


    //--------------------------------------------------------pod_auto_vector
    template<class T, unsigned Size> class pod_auto_vector
    {
    public:
        typedef T value_type;
        typedef pod_auto_vector<T, Size> self_type;

        pod_auto_vector() : m_size(0) {}

        void remove_all()            { m_size = 0; }
        void clear()                 { m_size = 0; }
        void add(const T& v)         { m_array[m_size++] = v; }
        void push_back(const T& v)   { m_array[m_size++] = v; }
        void inc_size(unsigned size) { m_size += size; }
        
        unsigned size() const { return m_size; }
        const T& operator [] (unsigned i) const { return m_array[i]; }
              T& operator [] (unsigned i)       { return m_array[i]; }
        const T& at(unsigned i) const           { return m_array[i]; }
              T& at(unsigned i)                 { return m_array[i]; }
        T  value_at(unsigned i) const           { return m_array[i]; }

    private:
        T m_array[Size];
        unsigned m_size;
    };


    //---------------------------------------------------------------pod_array
    template<class T> class pod_array
    {
    public:
        typedef T value_type;
        typedef pod_array<T> self_type;

        ~pod_array() { pod_allocator<T>::deallocate(m_array, m_size); }
        pod_array() : m_array(0), m_size(0) {}

        pod_array(unsigned size) : 
            m_array(pod_allocator<T>::allocate(size)), 
            m_size(size) 
        {}

        pod_array(const self_type& v) : 
            m_array(pod_allocator<T>::allocate(v.m_size)), 
            m_size(v.m_size) 
        {
            memcpy(m_array, v.m_array, sizeof(T) * m_size);
        }

        void resize(unsigned size)
        {
            if(size != m_size)
            {
                pod_allocator<T>::deallocate(m_array, m_size);
                m_array = pod_allocator<T>::allocate(m_size = size);
            }
        }
        const self_type& operator = (const self_type& v)
        {
            resize(v.size());
            memcpy(m_array, v.m_array, sizeof(T) * m_size);
            return *this;
        }

        unsigned size() const { return m_size; }
        const T& operator [] (unsigned i) const { return m_array[i]; }
              T& operator [] (unsigned i)       { return m_array[i]; }
        const T& at(unsigned i) const           { return m_array[i]; }
              T& at(unsigned i)                 { return m_array[i]; }
        T  value_at(unsigned i) const           { return m_array[i]; }

        const T* data() const { return m_array; }
              T* data()       { return m_array; }
    private:
        T*       m_array;
        unsigned m_size;
    };



    //--------------------------------------------------------------pod_vector
    // A simple class template to store Plain Old Data, a vector
    // of a fixed size. The data is continous in memory
    //------------------------------------------------------------------------
    template<class T> class pod_vector
    {
    public:
        typedef T value_type;

        ~pod_vector() { pod_allocator<T>::deallocate(m_array, m_capacity); }
        pod_vector() : m_size(0), m_capacity(0), m_array(0) {}
        pod_vector(unsigned cap, unsigned extra_tail=0);

        // Copying
        pod_vector(const pod_vector<T>&);
        const pod_vector<T>& operator = (const pod_vector<T>&);

        // Set new capacity. All data is lost, size is set to zero.
        void capacity(unsigned cap, unsigned extra_tail=0);
        unsigned capacity() const { return m_capacity; }

        // Allocate n elements. All data is lost, 
        // but elements can be accessed in range 0...size-1. 
        void allocate(unsigned size, unsigned extra_tail=0);

        // Resize keeping the content.
        void resize(unsigned new_size);

        void zero()
        {
            memset(m_array, 0, sizeof(T) * m_size);
        }

        void add(const T& v)         { m_array[m_size++] = v; }
        void push_back(const T& v)   { m_array[m_size++] = v; }
        void insert_at(unsigned pos, const T& val);
        void inc_size(unsigned size) { m_size += size; }
        unsigned size()      const   { return m_size; }
        unsigned byte_size() const   { return m_size * sizeof(T); }
        void serialize(int8u* ptr) const;
        void deserialize(const int8u* data, unsigned byte_size);
        const T& operator [] (unsigned i) const { return m_array[i]; }
              T& operator [] (unsigned i)       { return m_array[i]; }
        const T& at(unsigned i) const           { return m_array[i]; }
              T& at(unsigned i)                 { return m_array[i]; }
        T  value_at(unsigned i) const           { return m_array[i]; }

        const T* data() const { return m_array; }
              T* data()       { return m_array; }

        void remove_all()         { m_size = 0; }
        void clear()              { m_size = 0; }
        void cut_at(unsigned num) { if(num < m_size) m_size = num; }

    private:
        unsigned m_size;
        unsigned m_capacity;
        T*       m_array;
    };

    //------------------------------------------------------------------------
    template<class T> 
    void pod_vector<T>::capacity(unsigned cap, unsigned extra_tail)
    {
        m_size = 0;
        if(cap > m_capacity)
        {
            pod_allocator<T>::deallocate(m_array, m_capacity);
            m_capacity = cap + extra_tail;
            m_array = m_capacity ? pod_allocator<T>::allocate(m_capacity) : 0;
        }
    }

    //------------------------------------------------------------------------
    template<class T> 
    void pod_vector<T>::allocate(unsigned size, unsigned extra_tail)
    {
        capacity(size, extra_tail);
        m_size = size;
    }


    //------------------------------------------------------------------------
    template<class T> 
    void pod_vector<T>::resize(unsigned new_size)
    {
        if(new_size > m_size)
        {
            if(new_size > m_capacity)
            {
                T* data = pod_allocator<T>::allocate(new_size);
                memcpy(data, m_array, m_size * sizeof(T));
                pod_allocator<T>::deallocate(m_array, m_capacity);
                m_array = data;
            }
        }
        else
        {
            m_size = new_size;
        }
    }

    //------------------------------------------------------------------------
    template<class T> pod_vector<T>::pod_vector(unsigned cap, unsigned extra_tail) :
        m_size(0), 
        m_capacity(cap + extra_tail), 
        m_array(pod_allocator<T>::allocate(m_capacity)) {}

    //------------------------------------------------------------------------
    template<class T> pod_vector<T>::pod_vector(const pod_vector<T>& v) :
        m_size(v.m_size),
        m_capacity(v.m_capacity),
        m_array(v.m_capacity ? pod_allocator<T>::allocate(v.m_capacity) : 0)
    {
        memcpy(m_array, v.m_array, sizeof(T) * v.m_size);
    }

    //------------------------------------------------------------------------
    template<class T> const pod_vector<T>& 
    pod_vector<T>::operator = (const pod_vector<T>&v)
    {
        allocate(v.m_size);
        if(v.m_size) memcpy(m_array, v.m_array, sizeof(T) * v.m_size);
        return *this;
    }

    //------------------------------------------------------------------------
    template<class T> void pod_vector<T>::serialize(int8u* ptr) const
    { 
        if(m_size) memcpy(ptr, m_array, m_size * sizeof(T)); 
    }

    //------------------------------------------------------------------------
    template<class T> 
    void pod_vector<T>::deserialize(const int8u* data, unsigned byte_size)
    {
        byte_size /= sizeof(T);
        allocate(byte_size);
        if(byte_size) memcpy(m_array, data, byte_size * sizeof(T));
    }

    //------------------------------------------------------------------------
    template<class T> 
    void pod_vector<T>::insert_at(unsigned pos, const T& val)
    {
        if(pos >= m_size) 
        {
            m_array[m_size] = val;
        }
        else
        {
            memmove(m_array + pos + 1, m_array + pos, (m_size - pos) * sizeof(T));
            m_array[pos] = val;
        }
        ++m_size;
    }

    //---------------------------------------------------------------pod_bvector
    // A simple class template to store Plain Old Data, similar to std::deque
    // It doesn't reallocate memory but instead, uses blocks of data of size 
    // of (1 << S), that is, power of two. The data is NOT contiguous in memory, 
    // so the only valid access method is operator [] or curr(), prev(), next()
    // 
    // There reallocs occure only when the pool of pointers to blocks needs 
    // to be extended (it happens very rarely). You can control the value 
    // of increment to reallocate the pointer buffer. See the second constructor.
    // By default, the incremeent value equals (1 << S), i.e., the block size.
    //------------------------------------------------------------------------
    template<class T, unsigned S=6> class pod_bvector
    {
    public:
        enum block_scale_e
        {   
            block_shift = S,
            block_size  = 1 << block_shift,
            block_mask  = block_size - 1
        };

        typedef T value_type;

        ~pod_bvector();
        pod_bvector();
        pod_bvector(unsigned block_ptr_inc);

        // Copying
        pod_bvector(const pod_bvector<T, S>& v);
        const pod_bvector<T, S>& operator = (const pod_bvector<T, S>& v);

        void remove_all() { m_size = 0; }
        void clear()      { m_size = 0; }
        void free_all()   { free_tail(0); }
        void free_tail(unsigned size);
        void add(const T& val);
        void push_back(const T& val) { add(val); }
        void modify_last(const T& val);
        void remove_last();

        int allocate_continuous_block(unsigned num_elements);

        void add_array(const T* ptr, unsigned num_elem)
        {
            while(num_elem--)
            {
                add(*ptr++);
            }
        }

        template<class DataAccessor> void add_data(DataAccessor& data)
        {
            while(data.size())
            {
                add(*data);
                ++data;
            }
        }

        void cut_at(unsigned size)
        {
            if(size < m_size) m_size = size;
        }

        unsigned size() const { return m_size; }

        const T& operator [] (unsigned i) const
        {
            return m_blocks[i >> block_shift][i & block_mask];
        }

        T& operator [] (unsigned i)
        {
            return m_blocks[i >> block_shift][i & block_mask];
        }

        const T& at(unsigned i) const
        { 
            return m_blocks[i >> block_shift][i & block_mask];
        }

        T& at(unsigned i) 
        { 
            return m_blocks[i >> block_shift][i & block_mask];
        }

        T value_at(unsigned i) const
        { 
            return m_blocks[i >> block_shift][i & block_mask];
        }

        const T& curr(unsigned idx) const
        {
            return (*this)[idx];
        }

        T& curr(unsigned idx)
        {
            return (*this)[idx];
        }

        const T& prev(unsigned idx) const
        {
            return (*this)[(idx + m_size - 1) % m_size];
        }

        T& prev(unsigned idx)
        {
            return (*this)[(idx + m_size - 1) % m_size];
        }

        const T& next(unsigned idx) const
        {
            return (*this)[(idx + 1) % m_size];
        }

        T& next(unsigned idx)
        {
            return (*this)[(idx + 1) % m_size];
        }

        const T& last() const
        {
            return (*this)[m_size - 1];
        }

        T& last()
        {
            return (*this)[m_size - 1];
        }

        unsigned byte_size() const;
        void serialize(int8u* ptr) const;
        void deserialize(const int8u* data, unsigned byte_size);
        void deserialize(unsigned start, const T& empty_val, 
                         const int8u* data, unsigned byte_size);

        template<class ByteAccessor> 
        void deserialize(ByteAccessor data)
        {
            remove_all();
            unsigned elem_size = data.size() / sizeof(T);

            for(unsigned i = 0; i < elem_size; ++i)
            {
                int8u* ptr = (int8u*)data_ptr();
                for(unsigned j = 0; j < sizeof(T); ++j)
                {
                    *ptr++ = *data;
                    ++data;
                }
                ++m_size;
            }
        }

        template<class ByteAccessor>
        void deserialize(unsigned start, const T& empty_val, ByteAccessor data)
        {
            while(m_size < start)
            {
                add(empty_val);
            }

            unsigned elem_size = data.size() / sizeof(T);
            for(unsigned i = 0; i < elem_size; ++i)
            {
                int8u* ptr;
                if(start + i < m_size)
                {
                    ptr = (int8u*)(&((*this)[start + i]));
                }
                else
                {
                    ptr = (int8u*)data_ptr();
                    ++m_size;
                }
                for(unsigned j = 0; j < sizeof(T); ++j)
                {
                    *ptr++ = *data;
                    ++data;
                }
            }
        }

        const T* block(unsigned nb) const { return m_blocks[nb]; }

    private:
        void allocate_block(unsigned nb);
        T*   data_ptr();

        unsigned        m_size;
        unsigned        m_num_blocks;
        unsigned        m_max_blocks;
        T**             m_blocks;
        unsigned        m_block_ptr_inc;
    };


    //------------------------------------------------------------------------
    template<class T, unsigned S> pod_bvector<T, S>::~pod_bvector()
    {
        if(m_num_blocks)
        {
            T** blk = m_blocks + m_num_blocks - 1;
            while(m_num_blocks--)
            {
                pod_allocator<T>::deallocate(*blk, block_size);
                --blk;
            }
        }
        pod_allocator<T*>::deallocate(m_blocks, m_max_blocks);
    }


    //------------------------------------------------------------------------
    template<class T, unsigned S> 
    void pod_bvector<T, S>::free_tail(unsigned size)
    {
        if(size < m_size)
        {
            unsigned nb = (size + block_mask) >> block_shift;
            while(m_num_blocks > nb)
            {
                pod_allocator<T>::deallocate(m_blocks[--m_num_blocks], block_size);
            }
            if(m_num_blocks == 0)
            {
                pod_allocator<T*>::deallocate(m_blocks, m_max_blocks);
                m_blocks = 0;
                m_max_blocks = 0;
            }
            m_size = size;
        }
    }


    //------------------------------------------------------------------------
    template<class T, unsigned S> pod_bvector<T, S>::pod_bvector() :
        m_size(0),
        m_num_blocks(0),
        m_max_blocks(0),
        m_blocks(0),
        m_block_ptr_inc(block_size)
    {
    }


    //------------------------------------------------------------------------
    template<class T, unsigned S> 
    pod_bvector<T, S>::pod_bvector(unsigned block_ptr_inc) :
        m_size(0),
        m_num_blocks(0),
        m_max_blocks(0),
        m_blocks(0),
        m_block_ptr_inc(block_ptr_inc)
    {
    }


    //------------------------------------------------------------------------
    template<class T, unsigned S> 
    pod_bvector<T, S>::pod_bvector(const pod_bvector<T, S>& v) :
        m_size(v.m_size),
        m_num_blocks(v.m_num_blocks),
        m_max_blocks(v.m_max_blocks),
        m_blocks(v.m_max_blocks ? 
                 pod_allocator<T*>::allocate(v.m_max_blocks) : 
                 0),
        m_block_ptr_inc(v.m_block_ptr_inc)
    {
        unsigned i;
        for(i = 0; i < v.m_num_blocks; ++i)
        {
            m_blocks[i] = pod_allocator<T>::allocate(block_size);
            memcpy(m_blocks[i], v.m_blocks[i], block_size * sizeof(T));
        }
    }


    //------------------------------------------------------------------------
    template<class T, unsigned S> 
    const pod_bvector<T, S>& 
    pod_bvector<T, S>::operator = (const pod_bvector<T, S>& v)
    {
        unsigned i;
        for(i = m_num_blocks; i < v.m_num_blocks; ++i)
        {
            allocate_block(i);
        }
        for(i = 0; i < v.m_num_blocks; ++i)
        {
            memcpy(m_blocks[i], v.m_blocks[i], block_size * sizeof(T));
        }
        m_size = v.m_size;
        return *this;
    }


    //------------------------------------------------------------------------
    template<class T, unsigned S>
    void pod_bvector<T, S>::allocate_block(unsigned nb)
    {
        if(nb >= m_max_blocks) 
        {
            T** new_blocks = pod_allocator<T*>::allocate(m_max_blocks + m_block_ptr_inc);

            if(m_blocks)
            {
                memcpy(new_blocks, 
                       m_blocks, 
                       m_num_blocks * sizeof(T*));

                pod_allocator<T*>::deallocate(m_blocks, m_max_blocks);
            }
            m_blocks = new_blocks;
            m_max_blocks += m_block_ptr_inc;
        }
        m_blocks[nb] = pod_allocator<T>::allocate(block_size);
        m_num_blocks++;
    }



    //------------------------------------------------------------------------
    template<class T, unsigned S>
    inline T* pod_bvector<T, S>::data_ptr()
    {
        unsigned nb = m_size >> block_shift;
        if(nb >= m_num_blocks)
        {
            allocate_block(nb);
        }
        return m_blocks[nb] + (m_size & block_mask);
    }



    //------------------------------------------------------------------------
    template<class T, unsigned S> 
    inline void pod_bvector<T, S>::add(const T& val)
    {
        *data_ptr() = val;
        ++m_size;
    }


    //------------------------------------------------------------------------
    template<class T, unsigned S> 
    inline void pod_bvector<T, S>::remove_last()
    {
        if(m_size) --m_size;
    }


    //------------------------------------------------------------------------
    template<class T, unsigned S> 
    void pod_bvector<T, S>::modify_last(const T& val)
    {
        remove_last();
        add(val);
    }


    //------------------------------------------------------------------------
    template<class T, unsigned S> 
    int pod_bvector<T, S>::allocate_continuous_block(unsigned num_elements)
    {
        if(num_elements < block_size)
        {
            data_ptr(); // Allocate initial block if necessary
            unsigned rest = block_size - (m_size & block_mask);
            unsigned index;
            if(num_elements <= rest)
            {
                // The rest of the block is good, we can use it
                //-----------------
                index = m_size;
                m_size += num_elements;
                return index;
            }

            // New block
            //---------------
            m_size += rest;
            data_ptr();
            index = m_size;
            m_size += num_elements;
            return index;
        }
        return -1; // Impossible to allocate
    }


    //------------------------------------------------------------------------
    template<class T, unsigned S> 
    unsigned pod_bvector<T, S>::byte_size() const
    {
        return m_size * sizeof(T);
    }


    //------------------------------------------------------------------------
    template<class T, unsigned S> 
    void pod_bvector<T, S>::serialize(int8u* ptr) const
    {
        unsigned i;
        for(i = 0; i < m_size; i++)
        {
            memcpy(ptr, &(*this)[i], sizeof(T));
            ptr += sizeof(T);
        }
    }

    //------------------------------------------------------------------------
    template<class T, unsigned S> 
    void pod_bvector<T, S>::deserialize(const int8u* data, unsigned byte_size)
    {
        remove_all();
        byte_size /= sizeof(T);
        for(unsigned i = 0; i < byte_size; ++i)
        {
            T* ptr = data_ptr();
            memcpy(ptr, data, sizeof(T));
            ++m_size;
            data += sizeof(T);
        }
    }


    // Replace or add a number of elements starting from "start" position
    //------------------------------------------------------------------------
    template<class T, unsigned S> 
    void pod_bvector<T, S>::deserialize(unsigned start, const T& empty_val, 
                                        const int8u* data, unsigned byte_size)
    {
        while(m_size < start)
        {
            add(empty_val);
        }

        byte_size /= sizeof(T);
        for(unsigned i = 0; i < byte_size; ++i)
        {
            if(start + i < m_size)
            {
                memcpy(&((*this)[start + i]), data, sizeof(T));
            }
            else
            {
                T* ptr = data_ptr();
                memcpy(ptr, data, sizeof(T));
                ++m_size;
            }
            data += sizeof(T);
        }
    }


    //---------------------------------------------------------block_allocator
    // Allocator for arbitrary POD data. Most usable in different cache
    // systems for efficient memory allocations. 
    // Memory is allocated with blocks of fixed size ("block_size" in
    // the constructor). If required size exceeds the block size the allocator
    // creates a new block of the required size. However, the most efficient
    // use is when the average reqired size is much less than the block size. 
    //------------------------------------------------------------------------
    class block_allocator
    {
        struct block_type
        {
            int8u*   data;
            unsigned size;
        };

    public:
        void remove_all()
        {
            if(m_num_blocks)
            {
                block_type* blk = m_blocks + m_num_blocks - 1;
                while(m_num_blocks--)
                {
                    pod_allocator<int8u>::deallocate(blk->data, blk->size);
                    --blk;
                }
                pod_allocator<block_type>::deallocate(m_blocks, m_max_blocks);
            }
            m_num_blocks = 0;
            m_max_blocks = 0;
            m_blocks = 0;
            m_buf_ptr = 0;
            m_rest = 0;
        }

        ~block_allocator()
        {
            remove_all();
        }

        block_allocator(unsigned block_size, unsigned block_ptr_inc=256-8) :
            m_block_size(block_size),
            m_block_ptr_inc(block_ptr_inc),
            m_num_blocks(0),
            m_max_blocks(0),
            m_blocks(0),
            m_buf_ptr(0),
            m_rest(0)
        {
        }
       

        int8u* allocate(unsigned size, unsigned alignment=1)
        {
            if(size == 0) return 0;
            if(size <= m_rest)
            {
                int8u* ptr = m_buf_ptr;
                if(alignment > 1)
                {
                    unsigned align = 
                        (alignment - unsigned((size_t)ptr) % alignment) % alignment;

                    size += align;
                    ptr += align;
                    if(size <= m_rest)
                    {
                        m_rest -= size;
                        m_buf_ptr += size;
                        return ptr;
                    }
                    allocate_block(size);
                    return allocate(size - align, alignment);
                }
                m_rest -= size;
                m_buf_ptr += size;
                return ptr;
            }
            allocate_block(size + alignment - 1);
            return allocate(size, alignment);
        }


    private:
        void allocate_block(unsigned size)
        {
            if(size < m_block_size) size = m_block_size;
            if(m_num_blocks >= m_max_blocks) 
            {
                block_type* new_blocks = 
                    pod_allocator<block_type>::allocate(m_max_blocks + m_block_ptr_inc);

                if(m_blocks)
                {
                    memcpy(new_blocks, 
                           m_blocks, 
                           m_num_blocks * sizeof(block_type));
                    pod_allocator<block_type>::deallocate(m_blocks, m_max_blocks);
                }
                m_blocks = new_blocks;
                m_max_blocks += m_block_ptr_inc;
            }

            m_blocks[m_num_blocks].size = size;
            m_blocks[m_num_blocks].data = 
                m_buf_ptr =
                pod_allocator<int8u>::allocate(size);

            m_num_blocks++;
            m_rest = size;
        }

        unsigned    m_block_size;
        unsigned    m_block_ptr_inc;
        unsigned    m_num_blocks;
        unsigned    m_max_blocks;
        block_type* m_blocks;
        int8u*      m_buf_ptr;
        unsigned    m_rest;
    };








    //------------------------------------------------------------------------
    enum quick_sort_threshold_e
    {
        quick_sort_threshold = 9
    };

    
    //-----------------------------------------------------------swap_elements
    template<class T> inline void swap_elements(T& a, T& b)
    {
        T temp = a;
        a = b;
        b = temp;
    }


    //--------------------------------------------------------------quick_sort
    template<class Array, class Less>
    void quick_sort(Array& arr, Less less)
    {
        if(arr.size() < 2) return;

        typename Array::value_type* e1;
        typename Array::value_type* e2;

        int  stack[80];
        int* top = stack; 
        int  limit = arr.size();
        int  base = 0;

        for(;;)
        {
            int len = limit - base;

            int i;
            int j;
            int pivot;

            if(len > quick_sort_threshold)
            {
                // we use base + len/2 as the pivot
                pivot = base + len / 2;
                swap_elements(arr[base], arr[pivot]);

                i = base + 1;
                j = limit - 1;

                // now ensure that *i <= *base <= *j 
                e1 = &(arr[j]); 
                e2 = &(arr[i]);
                if(less(*e1, *e2)) swap_elements(*e1, *e2);

                e1 = &(arr[base]); 
                e2 = &(arr[i]);
                if(less(*e1, *e2)) swap_elements(*e1, *e2);

                e1 = &(arr[j]); 
                e2 = &(arr[base]);
                if(less(*e1, *e2)) swap_elements(*e1, *e2);

                for(;;)
                {
                    do i++; while( less(arr[i], arr[base]) );
                    do j--; while( less(arr[base], arr[j]) );

                    if( i > j )
                    {
                        break;
                    }

                    swap_elements(arr[i], arr[j]);
                }

                swap_elements(arr[base], arr[j]);

                // now, push the largest sub-array
                if(j - base > limit - i)
                {
                    top[0] = base;
                    top[1] = j;
                    base   = i;
                }
                else
                {
                    top[0] = i;
                    top[1] = limit;
                    limit  = j;
                }
                top += 2;
            }
            else
            {
                // the sub-array is small, perform insertion sort
                j = base;
                i = j + 1;

                for(; i < limit; j = i, i++)
                {
                    for(; less(*(e1 = &(arr[j + 1])), *(e2 = &(arr[j]))); j--)
                    {
                        swap_elements(*e1, *e2);
                        if(j == base)
                        {
                            break;
                        }
                    }
                }
                if(top > stack)
                {
                    top  -= 2;
                    base  = top[0];
                    limit = top[1];
                }
                else
                {
                    break;
                }
            }
        }
    }




    //------------------------------------------------------remove_duplicates
    // Remove duplicates from a sorted array. It doesn't cut the 
    // tail of the array, it just returns the number of remaining elements.
    //-----------------------------------------------------------------------
    template<class Array, class Equal>
    unsigned remove_duplicates(Array& arr, Equal equal)
    {
        if(arr.size() < 2) return arr.size();

        unsigned i, j;
        for(i = 1, j = 1; i < arr.size(); i++)
        {
            typename Array::value_type& e = arr[i];
            if(!equal(e, arr[i - 1]))
            {
                arr[j++] = e;
            }
        }
        return j;
    }

    //--------------------------------------------------------invert_container
    template<class Array> void invert_container(Array& arr)
    {
        int i = 0;
        int j = arr.size() - 1;
        while(i < j)
        {
            swap_elements(arr[i++], arr[j--]);
        }
    }

    //------------------------------------------------------binary_search_pos
    template<class Array, class Value, class Less>
    unsigned binary_search_pos(const Array& arr, const Value& val, Less less)
    {
        if(arr.size() == 0) return 0;

        unsigned beg = 0;
        unsigned end = arr.size() - 1;

        if(less(val, arr[0])) return 0;
        if(less(arr[end], val)) return end + 1;

        while(end - beg > 1)
        {
            unsigned mid = (end + beg) >> 1;
            if(less(val, arr[mid])) end = mid; 
            else                    beg = mid;
        }

        //if(beg <= 0 && less(val, arr[0])) return 0;
        //if(end >= arr.size() - 1 && less(arr[end], val)) ++end;

        return end;
    }

    //----------------------------------------------------------range_adaptor
    template<class Array> class range_adaptor
    {
    public:
        typedef typename Array::value_type value_type;

        range_adaptor(Array& array, unsigned start, unsigned size) :
            m_array(array), m_start(start), m_size(size)
        {}

        unsigned size() const { return m_size; }
        const value_type& operator [] (unsigned i) const { return m_array[m_start + i]; }
              value_type& operator [] (unsigned i)       { return m_array[m_start + i]; }
        const value_type& at(unsigned i) const           { return m_array[m_start + i]; }
              value_type& at(unsigned i)                 { return m_array[m_start + i]; }
        value_type  value_at(unsigned i) const           { return m_array[m_start + i]; }

    private:
        Array& m_array;
        unsigned m_start;
        unsigned m_size;
    };

    //---------------------------------------------------------------int_less
    inline bool int_less(int a, int b) { return a < b; }

    //------------------------------------------------------------int_greater
    inline bool int_greater(int a, int b) { return a > b; }

    //----------------------------------------------------------unsigned_less
    inline bool unsigned_less(unsigned a, unsigned b) { return a < b; }

    //-------------------------------------------------------unsigned_greater
    inline bool unsigned_greater(unsigned a, unsigned b) { return a > b; }
}

#endif
