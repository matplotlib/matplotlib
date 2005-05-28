#include "agg_scanline_bin.h"

namespace agg
{
  class scanline_bin
  {
  public:
    ~scanline_bin();
    scanline_bin();
    void reset(int min_x, int max_x);
    void add_cell(int x, unsigned);
    void add_span(int x, unsigned len, unsigned);
    void add_cells(int x, unsigned len, const void*);
    void finalize(int y);
    void reset_spans();
    int            y();
    unsigned       num_spans() const; 
  };  

  class scanline32_bin
  {
  public:
    ~scanline32_bin();
    scanline32_bin();
    void reset(int min_x, int max_x);
    void add_cell(int x, unsigned);
    void add_span(int x, unsigned len, unsigned);
    void add_cells(int x, unsigned len, const void*);
    void finalize(int y);
    void reset_spans();
    int            y();
    unsigned       num_spans() const; 
  };  

}

