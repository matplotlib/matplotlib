#include "agg_path_storage.h"

namespace agg
{
  class path_storage
  {
  public:
    ~path_storage();
    path_storage();
    path_storage(const path_storage& ps);
    
    void remove_all();
    
    unsigned last_vertex(double* vertex_x, double* vertex_y) const;
    unsigned prev_vertex(double* vertex_x, double* vertex_y) const;
    
    void rel_to_abs(double* vertex_x, double* vertex_y) const;
    
    void move_to(double x, double y);
    void move_rel(double dx, double dy);
    
    void line_to(double x, double y);
    void line_rel(double dx, double dy);
    
    void arc_to(double rx, double ry,
		double angle,
		bool large_arc_flag,
		bool sweep_flag,
		double x, double y);
    
    void arc_rel(double rx, double ry,
		 double angle,
		 bool large_arc_flag,
		 bool sweep_flag,
		 double dx, double dy);
    
    void curve3(double x_ctrl, double y_ctrl, 
		double x_to,   double y_to);
    
    void curve3_rel(double dx_ctrl, double dy_ctrl, 
		    double dx_to,   double dy_to);
    
    void curve3(double x_to, double y_to);
    
    void curve3_rel(double dx_to, double dy_to);
    
    void curve4(double x_ctrl1, double y_ctrl1, 
		double x_ctrl2, double y_ctrl2, 
		double x_to,    double y_to);
    
    void curve4_rel(double dx_ctrl1, double dy_ctrl1, 
		    double dx_ctrl2, double dy_ctrl2, 
		    double dx_to,    double dy_to);
    
    void curve4(double x_ctrl2, double y_ctrl2, 
		double x_to,    double y_to);
    
    void curve4_rel(double x_ctrl2, double y_ctrl2, 
		    double x_to,    double y_to);
    
    
    void end_poly(unsigned flags = path_flags_close);
    
    void close_polygon(unsigned flags = path_flags_none);
    void add_poly(const double* vertices, unsigned num, 
		  bool solid_path = false,
		  unsigned end_flags = path_flags_none);
    
    unsigned start_new_path();
    
    void copy_from(const path_storage& ps);
    unsigned total_vertices() const;
    unsigned vertex(unsigned idx, double* vertex_x, double* vertex_y) const;
    unsigned command(unsigned idx) const;
    void     rewind(unsigned path_id);
    unsigned vertex(double* vertex_x, double* vertex_y);
    
    unsigned arrange_orientations(unsigned path_id, path_flags_e new_orientation);
    void arrange_orientations_all_paths(path_flags_e new_orientation);
    
    void flip_x(double x1, double x2);
    void flip_y(double y1, double y2);
    void add_vertex(double x, double y, unsigned cmd);
    void modify_vertex(unsigned idx, double x, double y);
    void modify_command(unsigned idx, unsigned cmd);
  };
  
}

