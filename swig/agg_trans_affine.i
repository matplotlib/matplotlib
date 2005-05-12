%include "agg_typemaps.i"



// ** affine transformation **
namespace agg{
  %rename(as_vec6) trans_affine::store_to(double* array6) const;
  %rename(get_rotation) trans_affine::rotation() const;
  %rename(get_translation) trans_affine::translation(double* OUTPUT, double* OUTPUT) const;
  %rename(get_scaling) trans_affine::scaling(double* OUTPUT, double* OUTPUT) const;

  class trans_affine
  {
  public:
    trans_affine();
    trans_affine(double v0, double v1, double v2, double v3, double v4, double v5);
    trans_affine(const double *parl, const double *parl);
    trans_affine(double x1, double y1, double x2, double y2, 
    		 const double *parl);
    trans_affine(const double *parl, 
		 double x1, double y1, double x2, double y2);
    const trans_affine& parl_to_parl(const double *parl, 
				     const double *parl);
    const trans_affine& rect_to_parl(double x1, double y1, 
				     double x2, double y2, 
				     const double *parl);
    const trans_affine& parl_to_rect(const double *parl, 
				     double x1, double y1, 
				     double x2, double y2);
    const trans_affine& reset();
    const trans_affine& multiply(const trans_affine& m);
    const trans_affine& premultiply(const trans_affine& m);
    const trans_affine& invert();
    const trans_affine& flip_x();
    const trans_affine& flip_y();
    
    void store_to(double* array6) const;
    
    const trans_affine& load_from(const double *parl);
    const trans_affine& operator *= (const trans_affine& m);
    trans_affine operator * (const trans_affine& m);
    trans_affine operator ~ () const;
    bool operator == (const trans_affine& m) const;
    bool operator != (const trans_affine& m) const;
    
    void transform(double* INOUT, double* INOUT) const;
    void inverse_transform(double* INOUT, double* INOUT) const;
    double determinant() const;
    double scale() const;
    bool is_identity(double epsilon = agg::affine_epsilon) const;
    bool is_equal(const trans_affine& m, double epsilon = agg::affine_epsilon) const;
    //avoid swig name clashes with classes trans_affine_rotation, etc
    double rotation() const;
    void translation(double* OUTPUT, double* OUTPUT) const;
    void   scaling(double* OUTPUT, double* OUTPUT) const;
  }; 
  

  class trans_affine_rotation : public trans_affine
  {
  public:
    trans_affine_rotation(double a);
  };

  class trans_affine_scaling : public trans_affine
  {
  public:
    trans_affine_scaling(double sx, double sy);
    trans_affine_scaling(double s);
  };

  class trans_affine_translation : public trans_affine
  {
  public:
    trans_affine_translation(double tx, double ty);
  };

  class trans_affine_skewing : public trans_affine
  {
  public:
    trans_affine_skewing(double sx, double sy);
  };

}
