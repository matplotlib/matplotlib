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
//
// Adaptation for 32-bit screen coordinates has been sponsored by 
// Liberty Technology Systems, Inc., visit http://lib-sys.com
//
// Liberty Technology Systems, Inc. is the provider of
// PostScript and PDF technology for software developers.
// 
//----------------------------------------------------------------------------

#ifndef AGG_SCANLINE_STORAGE_AA_INCLUDED
#define AGG_SCANLINE_STORAGE_AA_INCLUDED

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "agg_array.h"


namespace agg
{

    //----------------------------------------------scanline_cell_storage
    template<class T> class scanline_cell_storage
    {
        struct extra_span
        {
            unsigned len;
            T*       ptr;
        };

    public:
        typedef T value_type;

        //---------------------------------------------------------------
        ~scanline_cell_storage()
        {
            remove_all();
        }

        //---------------------------------------------------------------
        scanline_cell_storage() :
            m_cells(128-2),
            m_extra_storage()
        {}


        // Copying
        //---------------------------------------------------------------
        scanline_cell_storage(const scanline_cell_storage<T>& v) :
            m_cells(v.m_cells),
            m_extra_storage()
        {
            copy_extra_storage(v);
        }

        //---------------------------------------------------------------
        const scanline_cell_storage<T>& 
        operator = (const scanline_cell_storage<T>& v)
        {
            remove_all();
            m_cells = v.m_cells;
            copy_extra_storage(v);
            return *this;
        }

        //---------------------------------------------------------------
        void remove_all()
        {
            int i;
            for(i = m_extra_storage.size()-1; i >= 0; --i)
            {
                pod_allocator<T>::deallocate(m_extra_storage[i].ptr,
                                             m_extra_storage[i].len);
            }
            m_extra_storage.remove_all();
            m_cells.remove_all();
        }

        //---------------------------------------------------------------
        int add_cells(const T* cells, unsigned num_cells)
        {
            int idx = m_cells.allocate_continuous_block(num_cells);
            if(idx >= 0)
            {
                T* ptr = &m_cells[idx];
                memcpy(ptr, cells, sizeof(T) * num_cells);
                return idx;
            }
            extra_span s;
            s.len = num_cells;
            s.ptr = pod_allocator<T>::allocate(num_cells);
            memcpy(s.ptr, cells, sizeof(T) * num_cells);
            m_extra_storage.add(s);
            return -int(m_extra_storage.size());
        }

        //---------------------------------------------------------------
        const T* operator [] (int idx) const
        {
            if(idx >= 0)
            {
                if((unsigned)idx >= m_cells.size()) return 0;
                return &m_cells[(unsigned)idx];
            }
            unsigned i = unsigned(-idx - 1);
            if(i >= m_extra_storage.size()) return 0;
            return m_extra_storage[i].ptr;
        }

        //---------------------------------------------------------------
        T* operator [] (int idx)
        {
            if(idx >= 0)
            {
                if((unsigned)idx >= m_cells.size()) return 0;
                return &m_cells[(unsigned)idx];
            }
            unsigned i = unsigned(-idx - 1);
            if(i >= m_extra_storage.size()) return 0;
            return m_extra_storage[i].ptr;
        }

    private:
        void copy_extra_storage(const scanline_cell_storage<T>& v)
        {
            unsigned i;
            for(i = 0; i < v.m_extra_storage.size(); ++i)
            {
                const extra_span& src = v.m_extra_storage[i];
                extra_span dst;
                dst.len = src.len;
                dst.ptr = pod_allocator<T>::allocate(dst.len);
                memcpy(dst.ptr, src.ptr, dst.len * sizeof(T));
                m_extra_storage.add(dst);
            }
        }

        pod_bvector<T, 12>         m_cells;
        pod_bvector<extra_span, 6> m_extra_storage;
    };






    //-----------------------------------------------scanline_storage_aa
    template<class T> class scanline_storage_aa
    {
    public:
        typedef T cover_type;

        //---------------------------------------------------------------
        struct span_data
        {
            int32 x;
            int32 len;       // If negative, it's a solid span, covers is valid
            int   covers_id; // The index of the cells in the scanline_cell_storage
        };

        //---------------------------------------------------------------
        struct scanline_data
        {
            int      y;
            unsigned num_spans;
            unsigned start_span;
        };


        //---------------------------------------------------------------
        class embedded_scanline
        {
        public:

            //-----------------------------------------------------------
            class const_iterator
            {
            public:
                struct span
                {
                    int32    x;
                    int32    len; // If negative, it's a solid span, covers is valid
                    const T* covers;
                };

                const_iterator() : m_storage(0) {}
                const_iterator(embedded_scanline& sl) :
                    m_storage(sl.m_storage),
                    m_span_idx(sl.m_scanline.start_span)
                {
                    init_span();
                }

                const span& operator*()  const { return m_span;  }
                const span* operator->() const { return &m_span; }

                void operator ++ ()
                {
                    ++m_span_idx;
                    init_span();
                }

            private:
                void init_span()
                {
                    const span_data& s = m_storage->span_by_index(m_span_idx);
                    m_span.x      = s.x;
                    m_span.len    = s.len;
                    m_span.covers = m_storage->covers_by_index(s.covers_id);
                }

                scanline_storage_aa* m_storage;
                unsigned                   m_span_idx;
                span                       m_span;
            };

            friend class const_iterator;


            //-----------------------------------------------------------
            embedded_scanline(const scanline_storage_aa& storage) :
                m_storage(&storage)
            {
                init(0);
            }

            //-----------------------------------------------------------
            void     reset(int, int)     {}
            unsigned num_spans()   const { return m_scanline.num_spans;  }
            int      y()           const { return m_scanline.y;          }
            const_iterator begin() const { return const_iterator(*this); }

            //-----------------------------------------------------------
            void init(unsigned scanline_idx)
            {
                m_scanline_idx = scanline_idx;
                m_scanline = m_storage->scanline_by_index(m_scanline_idx);
            }

        private:
            const scanline_storage_aa* m_storage;
            scanline_data              m_scanline;
            unsigned                   m_scanline_idx;
        };


        //---------------------------------------------------------------
        scanline_storage_aa() :
            m_covers(),
            m_spans(256-2),         // Block increment size
            m_scanlines(),
            m_min_x( 0x7FFFFFFF),
            m_min_y( 0x7FFFFFFF),
            m_max_x(-0x7FFFFFFF),
            m_max_y(-0x7FFFFFFF),
            m_cur_scanline(0)
        {
            m_fake_scanline.y = 0;
            m_fake_scanline.num_spans = 0;
            m_fake_scanline.start_span = 0;
            m_fake_span.x = 0;
            m_fake_span.len = 0;
            m_fake_span.covers_id = 0;
        }

        // Renderer Interface
        //---------------------------------------------------------------
        void prepare()
        {
            m_covers.remove_all();
            m_scanlines.remove_all();
            m_spans.remove_all();
            m_min_x =  0x7FFFFFFF;
            m_min_y =  0x7FFFFFFF;
            m_max_x = -0x7FFFFFFF;
            m_max_y = -0x7FFFFFFF;
            m_cur_scanline = 0;
        }

        //---------------------------------------------------------------
        template<class Scanline> void render(const Scanline& sl)
        {
            scanline_data sl_this;

            int y = sl.y();
            if(y < m_min_y) m_min_y = y;
            if(y > m_max_y) m_max_y = y;

            sl_this.y = y;
            sl_this.num_spans = sl.num_spans();
            sl_this.start_span = m_spans.size();
            typename Scanline::const_iterator span_iterator = sl.begin();

            unsigned num_spans = sl_this.num_spans;
            for(;;)
            {
                span_data sp;

                sp.x         = span_iterator->x;
                sp.len       = span_iterator->len;
                int len      = abs(int(sp.len));
                sp.covers_id = 
                    m_covers.add_cells(span_iterator->covers, 
                                       unsigned(len));
                m_spans.add(sp);
                int x1 = sp.x;
                int x2 = sp.x + len - 1;
                if(x1 < m_min_x) m_min_x = x1;
                if(x2 > m_max_x) m_max_x = x2;
                if(--num_spans == 0) break;
                ++span_iterator;
            }
            m_scanlines.add(sl_this);
        }


        //---------------------------------------------------------------
        // Iterate scanlines interface
        int min_x() const { return m_min_x; }
        int min_y() const { return m_min_y; }
        int max_x() const { return m_max_x; }
        int max_y() const { return m_max_y; }

        //---------------------------------------------------------------
        bool rewind_scanlines()
        {
            m_cur_scanline = 0;
            return m_scanlines.size() > 0;
        }


        //---------------------------------------------------------------
        template<class Scanline> bool sweep_scanline(Scanline& sl)
        {
            sl.reset_spans();
            for(;;)
            {
                if(m_cur_scanline >= m_scanlines.size()) return false;
                const scanline_data& sl_this = m_scanlines[m_cur_scanline];

                unsigned num_spans = sl_this.num_spans;
                unsigned span_idx  = sl_this.start_span;
                do
                {
                    const span_data& sp = m_spans[span_idx++];
                    const T* covers = covers_by_index(sp.covers_id);
                    if(sp.len < 0)
                    {
                        sl.add_span(sp.x, unsigned(-sp.len), *covers);
                    }
                    else
                    {
                        sl.add_cells(sp.x, sp.len, covers);
                    }
                }
                while(--num_spans);
                ++m_cur_scanline;
                if(sl.num_spans())
                {
                    sl.finalize(sl_this.y);
                    break;
                }
            }
            return true;
        }


        //---------------------------------------------------------------
        // Specialization for embedded_scanline
        bool sweep_scanline(embedded_scanline& sl)
        {
            do
            {
                if(m_cur_scanline >= m_scanlines.size()) return false;
                sl.init(m_cur_scanline);
                ++m_cur_scanline;
            }
            while(sl.num_spans() == 0);
            return true;
        }

        //---------------------------------------------------------------
        unsigned byte_size() const
        {
            unsigned i;
            unsigned size = sizeof(int32) * 4; // min_x, min_y, max_x, max_y

            for(i = 0; i < m_scanlines.size(); ++i)
            {
                size += sizeof(int32) * 3; // scanline size in bytes, Y, num_spans

                const scanline_data& sl_this = m_scanlines[i];

                unsigned num_spans = sl_this.num_spans;
                unsigned span_idx  = sl_this.start_span;
                do
                {
                    const span_data& sp = m_spans[span_idx++];

                    size += sizeof(int32) * 2;                // X, span_len
                    if(sp.len < 0)
                    {
                        size += sizeof(T);                    // cover
                    }
                    else
                    {
                        size += sizeof(T) * unsigned(sp.len); // covers
                    }
                }
                while(--num_spans);
            }
            return size;
        }


        //---------------------------------------------------------------
        static void write_int32(int8u* dst, int32 val)
        {
            dst[0] = ((const int8u*)&val)[0];
            dst[1] = ((const int8u*)&val)[1];
            dst[2] = ((const int8u*)&val)[2];
            dst[3] = ((const int8u*)&val)[3];
        }


        //---------------------------------------------------------------
        void serialize(int8u* data) const
        {
            unsigned i;

            write_int32(data, min_x()); // min_x
            data += sizeof(int32);
            write_int32(data, min_y()); // min_y
            data += sizeof(int32);
            write_int32(data, max_x()); // max_x
            data += sizeof(int32);
            write_int32(data, max_y()); // max_y
            data += sizeof(int32);

            for(i = 0; i < m_scanlines.size(); ++i)
            {
                const scanline_data& sl_this = m_scanlines[i];
                
                int8u* size_ptr = data;
                data += sizeof(int32);  // Reserve space for scanline size in bytes

                write_int32(data, sl_this.y);            // Y
                data += sizeof(int32);

                write_int32(data, sl_this.num_spans);    // num_spans
                data += sizeof(int32);

                unsigned num_spans = sl_this.num_spans;
                unsigned span_idx  = sl_this.start_span;
                do
                {
                    const span_data& sp = m_spans[span_idx++];
                    const T* covers = covers_by_index(sp.covers_id);

                    write_int32(data, sp.x);            // X
                    data += sizeof(int32);

                    write_int32(data, sp.len);          // span_len
                    data += sizeof(int32);

                    if(sp.len < 0)
                    {
                        memcpy(data, covers, sizeof(T));
                        data += sizeof(T);
                    }
                    else
                    {
                        memcpy(data, covers, unsigned(sp.len) * sizeof(T));
                        data += sizeof(T) * unsigned(sp.len);
                    }
                }
                while(--num_spans);
                write_int32(size_ptr, int32(unsigned(data - size_ptr)));
            }
        }


        //---------------------------------------------------------------
        const scanline_data& scanline_by_index(unsigned i) const
        {
            return (i < m_scanlines.size()) ? m_scanlines[i] : m_fake_scanline;
        }

        //---------------------------------------------------------------
        const span_data& span_by_index(unsigned i) const
        {
            return (i < m_spans.size()) ? m_spans[i] : m_fake_span;
        }

        //---------------------------------------------------------------
        const T* covers_by_index(int i) const
        {
            return m_covers[i];
        }

    private:
        scanline_cell_storage<T>      m_covers;
        pod_bvector<span_data, 10>    m_spans;
        pod_bvector<scanline_data, 8> m_scanlines;
        span_data     m_fake_span;
        scanline_data m_fake_scanline;
        int           m_min_x;
        int           m_min_y;
        int           m_max_x;
        int           m_max_y;
        unsigned      m_cur_scanline;
    };


    typedef scanline_storage_aa<int8u>  scanline_storage_aa8;  //--------scanline_storage_aa8
    typedef scanline_storage_aa<int16u> scanline_storage_aa16; //--------scanline_storage_aa16
    typedef scanline_storage_aa<int32u> scanline_storage_aa32; //--------scanline_storage_aa32




    //------------------------------------------serialized_scanlines_adaptor_aa
    template<class T> class serialized_scanlines_adaptor_aa
    {
    public:
        typedef T cover_type;

        //---------------------------------------------------------------------
        class embedded_scanline
        {
        public:
            typedef T cover_type;

            //-----------------------------------------------------------------
            class const_iterator
            {
            public:
                struct span
                {
                    int32    x;
                    int32    len; // If negative, it's a solid span, "covers" is valid
                    const T* covers; 
                };

                const_iterator() : m_ptr(0) {}
                const_iterator(const embedded_scanline* sl) :
                    m_ptr(sl->m_ptr),
                    m_dx(sl->m_dx)
                {
                    init_span();
                }

                const span& operator*()  const { return m_span;  }
                const span* operator->() const { return &m_span; }

                void operator ++ ()
                {
                    if(m_span.len < 0) 
                    {
                        m_ptr += sizeof(T);
                    }
                    else 
                    {
                        m_ptr += m_span.len * sizeof(T);
                    }
                    init_span();
                }

            private:
                int read_int32()
                {
                    int32 val;
                    ((int8u*)&val)[0] = *m_ptr++;
                    ((int8u*)&val)[1] = *m_ptr++;
                    ((int8u*)&val)[2] = *m_ptr++;
                    ((int8u*)&val)[3] = *m_ptr++;
                    return val;
                }

                void init_span()
                {
                    m_span.x      = read_int32() + m_dx;
                    m_span.len    = read_int32();
                    m_span.covers = m_ptr;
                }

                const int8u* m_ptr;
                span         m_span;
                int          m_dx;
            };

            friend class const_iterator;


            //-----------------------------------------------------------------
            embedded_scanline() : m_ptr(0), m_y(0), m_num_spans(0) {}

            //-----------------------------------------------------------------
            void     reset(int, int)     {}
            unsigned num_spans()   const { return m_num_spans;  }
            int      y()           const { return m_y;          }
            const_iterator begin() const { return const_iterator(this); }


        private:
            //-----------------------------------------------------------------
            int read_int32()
            {
                int32 val;
                ((int8u*)&val)[0] = *m_ptr++;
                ((int8u*)&val)[1] = *m_ptr++;
                ((int8u*)&val)[2] = *m_ptr++;
                ((int8u*)&val)[3] = *m_ptr++;
                return val;
            }

        public:
            //-----------------------------------------------------------------
            void init(const int8u* ptr, int dx, int dy)
            {
                m_ptr       = ptr;
                m_y         = read_int32() + dy;
                m_num_spans = unsigned(read_int32());
                m_dx        = dx;
            }

        private:
            const int8u* m_ptr;
            int          m_y;
            unsigned     m_num_spans;
            int          m_dx;
        };



    public:
        //--------------------------------------------------------------------
        serialized_scanlines_adaptor_aa() :
            m_data(0),
            m_end(0),
            m_ptr(0),
            m_dx(0),
            m_dy(0),
            m_min_x(0x7FFFFFFF),
            m_min_y(0x7FFFFFFF),
            m_max_x(-0x7FFFFFFF),
            m_max_y(-0x7FFFFFFF)
        {}

        //--------------------------------------------------------------------
        serialized_scanlines_adaptor_aa(const int8u* data, unsigned size,
                                        double dx, double dy) :
            m_data(data),
            m_end(data + size),
            m_ptr(data),
            m_dx(iround(dx)),
            m_dy(iround(dy)),
            m_min_x(0x7FFFFFFF),
            m_min_y(0x7FFFFFFF),
            m_max_x(-0x7FFFFFFF),
            m_max_y(-0x7FFFFFFF)
        {}

        //--------------------------------------------------------------------
        void init(const int8u* data, unsigned size, double dx, double dy)
        {
            m_data  = data;
            m_end   = data + size;
            m_ptr   = data;
            m_dx    = iround(dx);
            m_dy    = iround(dy);
            m_min_x = 0x7FFFFFFF;
            m_min_y = 0x7FFFFFFF;
            m_max_x = -0x7FFFFFFF;
            m_max_y = -0x7FFFFFFF;
        }

    private:
        //--------------------------------------------------------------------
        int read_int32()
        {
            int32 val;
            ((int8u*)&val)[0] = *m_ptr++;
            ((int8u*)&val)[1] = *m_ptr++;
            ((int8u*)&val)[2] = *m_ptr++;
            ((int8u*)&val)[3] = *m_ptr++;
            return val;
        }

        //--------------------------------------------------------------------
        unsigned read_int32u()
        {
            int32u val;
            ((int8u*)&val)[0] = *m_ptr++;
            ((int8u*)&val)[1] = *m_ptr++;
            ((int8u*)&val)[2] = *m_ptr++;
            ((int8u*)&val)[3] = *m_ptr++;
            return val;
        }
        
    public:
        // Iterate scanlines interface
        //--------------------------------------------------------------------
        bool rewind_scanlines()
        {
            m_ptr = m_data;
            if(m_ptr < m_end)
            {
                m_min_x = read_int32u() + m_dx;
                m_min_y = read_int32u() + m_dy;
                m_max_x = read_int32u() + m_dx;
                m_max_y = read_int32u() + m_dy;
            }
            return m_ptr < m_end;
        }

        //--------------------------------------------------------------------
        int min_x() const { return m_min_x; }
        int min_y() const { return m_min_y; }
        int max_x() const { return m_max_x; }
        int max_y() const { return m_max_y; }

        //--------------------------------------------------------------------
        template<class Scanline> bool sweep_scanline(Scanline& sl)
        {
            sl.reset_spans();
            for(;;)
            {
                if(m_ptr >= m_end) return false;

                read_int32();      // Skip scanline size in bytes
                int y = read_int32() + m_dy;
                unsigned num_spans = read_int32();

                do
                {
                    int x = read_int32() + m_dx;
                    int len = read_int32();

                    if(len < 0)
                    {
                        sl.add_span(x, unsigned(-len), *m_ptr);
                        m_ptr += sizeof(T);
                    }
                    else
                    {
                        sl.add_cells(x, len, m_ptr);
                        m_ptr += len * sizeof(T);
                    }
                }
                while(--num_spans);

                if(sl.num_spans())
                {
                    sl.finalize(y);
                    break;
                }
            }
            return true;
        }


        //--------------------------------------------------------------------
        // Specialization for embedded_scanline
        bool sweep_scanline(embedded_scanline& sl)
        {
            do
            {
                if(m_ptr >= m_end) return false;

                unsigned byte_size = read_int32u();
                sl.init(m_ptr, m_dx, m_dy);
                m_ptr += byte_size - sizeof(int32);
            }
            while(sl.num_spans() == 0);
            return true;
        }

    private:
        const int8u* m_data;
        const int8u* m_end;
        const int8u* m_ptr;
        int          m_dx;
        int          m_dy;
        int          m_min_x;
        int          m_min_y;
        int          m_max_x;
        int          m_max_y;
    };



    typedef serialized_scanlines_adaptor_aa<int8u>  serialized_scanlines_adaptor_aa8;  //----serialized_scanlines_adaptor_aa8
    typedef serialized_scanlines_adaptor_aa<int16u> serialized_scanlines_adaptor_aa16; //----serialized_scanlines_adaptor_aa16
    typedef serialized_scanlines_adaptor_aa<int32u> serialized_scanlines_adaptor_aa32; //----serialized_scanlines_adaptor_aa32

}


#endif

