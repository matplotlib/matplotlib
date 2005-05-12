#include "agg_rasterizer_scanline_aa.h"
%rename(rasterizer_scanline_aa) rasterizer_scanline_aa<>;
class rasterizer_scanline_aa<>
{

public:

  rasterizer_scanline_aa<>();
  void reset(); 
  void filling_rule(agg::filling_rule_e filling_rule);
  void clip_box(double x1, double y1, double x2, double y2);
  void reset_clipping();
  //template<class GammaF> void gamma(const GammaF& gamma_function)
  unsigned apply_gamma(unsigned cover) const; 
  void add_vertex(double x, double y, unsigned cmd);
  void move_to(int x, int y);
  void line_to(int x, int y);
  void close_polygon();
  void move_to_d(double x, double y);
  void line_to_d(double x, double y);
  int min_x() const;
  int min_y() const;
  int max_x() const;
  int max_y() const;
  
  unsigned calculate_alpha(int area);
  void sort();
  bool rewind_scanlines();
  //template<class Scanline> bool sweep_scanline(Scanline& sl);
  bool hit_test(int tx, int ty);
  //fixme void add_xy(const double* x, const double* y, unsigned n);
  //template<class VertexSource>

  void add_path(path_t& vs, unsigned id=0);
  void add_path(stroke_t& vs, unsigned id=0);

  void add_path(transpath_t& vs, unsigned id=0);
  void add_path(stroketrans_t& vs, unsigned id=0);

  void add_path(curve_t& vs, unsigned id=0);
  void add_path(strokecurve_t& vs, unsigned id=0);

  void add_path(transcurve_t& vs, unsigned id=0);
  void add_path(stroketranscurve_t& vs, unsigned id=0);

  void add_path(curvetrans_t& vs, unsigned id=0);
  void add_path(strokecurvetrans_t& vs, unsigned id=0);

};
