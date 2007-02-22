// File: _wxagg.cpp
// Purpose: Accelerate WXAgg by doing the agg->wxWidgets conversions in C++.
// Author: Ken McIvor <mcivor@iit.edu>
//
// Copyright 2005 Illinois Institute of Technology
// Derived from `_gtkagg.cpp', Copyright 2004-2005 John Hunter
//
// See the file "LICENSE" for information on usage and redistribution
// of this file, and for a DISCLAIMER OF ALL WARRANTIES.


// TODO:
// * Better type checking.
//
// * Make the `bbox' argument optional.
//
// * Determine if there are any thread-safety issues with this implementation.
//
// * Perform some AGG kung-fu to let us slice a region out of a
//   rendering_buffer and convert it from RGBA to RGB on the fly, rather than
//   making itermediate copies.  This could be of use in _gtkagg and _tkagg as
//   well.
//
// * Write an agg_to_wx_bitmap() that works more like agg_to_gtk_drawable(),
//   drawing directly to a bitmap.
//
//   This was the initial plan, except that I had not idea how to take a
//   wx.Bitmap Python shadow class and turn it into a wxBitmap pointer.
//
//   It appears that this is the way to do it:
//       bool success = wxPyConvertSwigPtr(pyBitmap, (void**)&bitmap,
//           _T("wxBitmap"));
//
//   I'm not sure this will speed things up much, since wxWidgets requires you
//   to go AGG->wx.Image->wx.Bitmap before you can blit using a MemoryDC.


#include <cstring>
#include <cerrno>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <utility>
#include <fstream>
#include <stdlib.h>

#include "agg_basics.h"
#include "_backend_agg.h"
#include "_transforms.h"
#include "agg_pixfmt_rgba.h"
#include "util/agg_color_conv_rgb8.h"

#include <wx/image.h>
#include <wx/bitmap.h>
#include <wx/wxPython/wxPython.h>


// forward declarations
static wxImage  *convert_agg2image(RendererAgg *aggRenderer, Bbox *clipbox);
static wxBitmap *convert_agg2bitmap(RendererAgg *aggRenderer, Bbox *clipbox);


// the extension module
class _wxagg_module : public Py::ExtensionModule<_wxagg_module>
{
public:

  _wxagg_module()
    : Py::ExtensionModule<_wxagg_module>( "_wxkagg" )
  {
    add_varargs_method("convert_agg_to_wx_image",
        &_wxagg_module::convert_agg_to_wx_image,
        "Convert the region of the agg buffer bounded by bbox to a wx.Image."
        "  If bbox\nis None, the entire buffer is converted.\n\nNote: agg must"
        " be a backend_agg.RendererAgg instance.");

    add_varargs_method("convert_agg_to_wx_bitmap",
        &_wxagg_module::convert_agg_to_wx_bitmap,
        "Convert the region of the agg buffer bounded by bbox to a wx.Bitmap."
        "  If bbox\nis None, the entire buffer is converted.\n\nNote: agg must"
        " be a backend_agg.RendererAgg instance.");

    initialize( "The _wxagg module" );
  }

  virtual ~_wxagg_module() {}

private:

  Py::Object convert_agg_to_wx_image(const Py::Tuple &args)
  {
    args.verify_length(2);

    RendererAgg* aggRenderer = static_cast<RendererAgg*>(
        args[0].getAttr("_renderer").ptr());

    Bbox *clipbox = NULL;
    if (args[1].ptr() != Py_None)
        clipbox = static_cast<Bbox*>(args[1].ptr());

    // convert the buffer
    wxImage *image = convert_agg2image(aggRenderer, clipbox);

    // wrap a wx.Image around the result and return it
    PyObject *pyWxImage = wxPyConstructObject(image, _T("wxImage"), 1);
    if (pyWxImage == NULL) {
        throw Py::MemoryError(
            "_wxagg.convert_agg_to_wx_image(): could not create the wx.Image");
    }

    return Py::asObject(pyWxImage);
  }


  Py::Object convert_agg_to_wx_bitmap(const Py::Tuple &args) {
    args.verify_length(2);

    RendererAgg* aggRenderer = static_cast<RendererAgg*>(
       args[0].getAttr("_renderer").ptr());

    Bbox *clipbox = NULL;
    if (args[1].ptr() != Py_None)
        clipbox = static_cast<Bbox*>(args[1].ptr());

    // convert the buffer
    wxBitmap *bitmap = convert_agg2bitmap(aggRenderer, clipbox);

    // wrap a wx.Bitmap around the result and return it
    PyObject *pyWxBitmap = wxPyConstructObject(bitmap, _T("wxBitmap"), 1);
    if (pyWxBitmap == NULL) {
        throw Py::MemoryError(
          "_wxagg.convert_agg_to_wx_bitmap(): could not create the wx.Bitmap");
    }

    return Py::asObject(pyWxBitmap);
  }
};


//
// Implementation Functions
//

static wxImage *convert_agg2image(RendererAgg *aggRenderer, Bbox *clipbox)
{
    int srcWidth  = 1;
    int srcHeight = 1;
    int srcStride = 1;

    bool deleteSrcBuffer = false;
    agg::int8u *srcBuffer = NULL;

    if (clipbox == NULL) {
        // Convert everything: rgba => rgb -> image
        srcBuffer = aggRenderer->pixBuffer;
        srcWidth  = (int) aggRenderer->get_width();
        srcHeight = (int) aggRenderer->get_height();
        srcStride = (int) aggRenderer->get_width()*4;
    } else {
        // Convert a region: rgba => clipped rgba => rgb -> image
        double l = clipbox->ll_api()->x_api()->val() ;
        double b = clipbox->ll_api()->y_api()->val();
        double r = clipbox->ur_api()->x_api()->val() ;
        double t = clipbox->ur_api()->y_api()->val() ;

        srcWidth = (int) (r-l);
        srcHeight = (int) (t-b);
        srcStride = srcWidth*4;

        deleteSrcBuffer = true;
        srcBuffer = new agg::int8u[srcStride*srcHeight];
        if (srcBuffer == NULL) {
            throw Py::MemoryError(
                "_wxagg::convert_agg2image(): could not allocate `srcBuffer'");
        }

        int h = (int) aggRenderer->get_height();
        agg::rect_base<int> region(
            (int) l,      // x1
            h - (int) t,  // y1
            (int) r,      // x2
            h - (int) b); // y2

        agg::rendering_buffer rbuf;
        rbuf.attach(srcBuffer, srcWidth, srcHeight, srcStride);
        pixfmt pf(rbuf);
        renderer_base rndr(pf);
        rndr.copy_from(*aggRenderer->renderingBuffer, &region,
            (int)-l, (int)(t-h));
    }

    // allocate the RGB data array

    // use malloc(3) because wxImage will use free(3)
    agg::int8u *destBuffer = (agg::int8u *) malloc(
        sizeof(agg::int8u)*srcWidth*3*srcHeight);

    if (destBuffer == NULL) {
        if (deleteSrcBuffer)
            delete [] srcBuffer;

        throw Py::MemoryError(
            "_wxagg::convert_agg2image(): could not allocate `destBuffer'");
    }

    // convert from RGBA to RGB
    agg::rendering_buffer rbSource;
    rbSource.attach(srcBuffer, srcWidth, srcHeight, srcStride);

    agg::rendering_buffer rbDest;
    rbDest.attach(destBuffer, srcWidth, srcHeight, srcWidth*3);

    agg::color_conv(&rbDest, &rbSource, agg::color_conv_rgba32_to_rgb24());

    // Create a wxImage using the RGB data
    wxImage *image = new wxImage(srcWidth, srcHeight, destBuffer);
    if (image == NULL) {
        if (deleteSrcBuffer)
            delete [] srcBuffer;

        free(destBuffer);
        throw Py::MemoryError(
            "_wxagg::convert_agg2image(): could not allocate `image'");
    }

    if (deleteSrcBuffer)
        delete [] srcBuffer;

    return image;
}


static wxBitmap *convert_agg2bitmap(RendererAgg *aggRenderer, Bbox *clipbox)
{
    // Convert everything: rgba => rgb -> image => bitmap
    // Convert a region:   rgba => clipped rgba => rgb -> image => bitmap
    wxImage *image = convert_agg2image(aggRenderer, clipbox);
    wxBitmap *bitmap = new wxBitmap(*image);

    image->Destroy();
    delete image;

    if (bitmap == NULL) {
        throw Py::MemoryError(
            "_wxagg::convert_agg2bitmap(): could not allocate `bitmap'");
    }

    return bitmap;
}


//
// Module Initialization
//

extern "C"
DL_EXPORT(void)
  init_wxagg(void)
{
  wxPyCoreAPI_IMPORT();
  //suppress an unused variable warning by creating _wxagg_module in two lines
  static _wxagg_module* _wxagg = NULL;
  _wxagg = new _wxagg_module;
};
