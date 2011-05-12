/* -*- mode: c++; c-basic-offset: 4 -*- */

/* image.h
 *
 */

#ifndef _IMAGE_H
#define _IMAGE_H
#include <utility>
#include "Python.h"

#include "agg_trans_affine.h"
#include "agg_rendering_buffer.h"
#include "agg_color_rgba.h"
#include "CXX/Extensions.hxx"



class Image : public Py::PythonExtension<Image>
{
public:
    Image();
    virtual ~Image();

    static void init_type(void);
    int setattr(const char*, const Py::Object &);
    Py::Object getattr(const char * name);

    Py::Object apply_rotation(const Py::Tuple& args);
    Py::Object apply_scaling(const Py::Tuple& args);
    Py::Object apply_translation(const Py::Tuple& args);
    Py::Object as_rgba_str(const Py::Tuple& args, const Py::Dict& kwargs);
    Py::Object color_conv(const Py::Tuple& args);
    Py::Object buffer_rgba(const Py::Tuple& args);
    Py::Object reset_matrix(const Py::Tuple& args);
    Py::Object get_matrix(const Py::Tuple& args);
    Py::Object resize(const Py::Tuple& args, const Py::Dict& kwargs);
    Py::Object get_aspect(const Py::Tuple& args);
    Py::Object get_size(const Py::Tuple& args);
    Py::Object get_size_out(const Py::Tuple& args);
    Py::Object get_interpolation(const Py::Tuple& args);
    Py::Object set_interpolation(const Py::Tuple& args);
    Py::Object set_aspect(const Py::Tuple& args);
    Py::Object set_bg(const Py::Tuple& args);
    inline Py::Object flipud_out(const Py::Tuple& args)
    {
        args.verify_length(0);
        int stride = rbufOut->stride();
        //std::cout << "flip before: " << rbufOut->stride() << std::endl;
        rbufOut->attach(bufferOut, colsOut, rowsOut, -stride);
        //std::cout << "flip after: " << rbufOut->stride() << std::endl;
        return Py::Object();
    }

    Py::Object flipud_in(const Py::Tuple& args);
    Py::Object set_resample(const Py::Tuple& args);
    Py::Object get_resample(const Py::Tuple& args);


    std::pair<agg::int8u*, bool> _get_output_buffer();
    enum {NEAREST,
          BILINEAR,
          BICUBIC,
          SPLINE16,
          SPLINE36,
          HANNING,
          HAMMING,
          HERMITE,
          KAISER,
          QUADRIC,
          CATROM,
          GAUSSIAN,
          BESSEL,
          MITCHELL,
          SINC,
          LANCZOS,
          BLACKMAN
         };

    //enum { BICUBIC=0, BILINEAR, BLACKMAN100, BLACKMAN256, BLACKMAN64,
    //   NEAREST, SINC144, SINC256, SINC64, SPLINE16, SPLINE36};
    enum { ASPECT_PRESERVE = 0, ASPECT_FREE};

    agg::int8u *bufferIn;
    agg::rendering_buffer *rbufIn;
    size_t colsIn, rowsIn;

    agg::int8u *bufferOut;
    agg::rendering_buffer *rbufOut;
    size_t colsOut, rowsOut;
    unsigned BPP;

    unsigned interpolation, aspect;
    agg::rgba bg;
    bool resample;
private:
    Py::Dict __dict__;
    agg::trans_affine srcMatrix, imageMatrix;

    static char apply_rotation__doc__[];
    static char apply_scaling__doc__[];
    static char apply_translation__doc__[];
    static char as_rgba_str__doc__[];
    static char color_conv__doc__[];
    static char buffer_rgba__doc__[];
    static char reset_matrix__doc__[];
    static char get_matrix__doc__[];
    static char resize__doc__[];
    static char get_aspect__doc__[];
    static char get_size__doc__[];
    static char get_size_out__doc__[];
    static char get_interpolation__doc__[];
    static char set_interpolation__doc__[];
    static char set_aspect__doc__[];
    static char set_bg__doc__[];
    static char flipud_out__doc__[];
    static char flipud_in__doc__[];
    static char get_resample__doc__[];
    static char set_resample__doc__[];

};


/*
class ImageComposite : public Py::PythonExtension<ImageComposite> {

}
*/


// the extension module
class _image_module : public Py::ExtensionModule<_image_module>
{
public:
    _image_module() : Py::ExtensionModule<_image_module>("_image")
    {
        Image::init_type();

        add_varargs_method("fromarray", &_image_module::fromarray,
                           "fromarray");
        add_varargs_method("fromarray2", &_image_module::fromarray2,
                           "fromarray2");
        add_varargs_method("frombyte", &_image_module::frombyte,
                           "frombyte");
        add_varargs_method("frombuffer", &_image_module::frombuffer,
                           "frombuffer");
        add_varargs_method("from_images", &_image_module::from_images,
                           "from_images");
        add_varargs_method("pcolor", &_image_module::pcolor,
                           "pcolor");
        add_varargs_method("pcolor2", &_image_module::pcolor2,
                           "pcolor2");
        initialize("The _image module");
    }

    ~_image_module() {}

private:
    Py::Object frombyte(const Py::Tuple &args);
    Py::Object frombuffer(const Py::Tuple &args);
    Py::Object fromarray(const Py::Tuple &args);
    Py::Object fromarray2(const Py::Tuple &args);
    Py::Object pcolor(const Py::Tuple &args);
    Py::Object pcolor2(const Py::Tuple &args);
    Py::Object from_images(const Py::Tuple &args);

    static char _image_module_fromarray__doc__[];
    static char _image_module_pcolor__doc__[];
    static char _image_module_pcolor2__doc__[];
    static char _image_module_fromarray2__doc__[];
    static char _image_module_frombyte__doc__[];
    static char _image_module_frombuffer__doc__[];
};



#endif

