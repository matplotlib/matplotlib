/***************************************************************************/
/*                                                                         */
/*  afloader.c                                                             */
/*                                                                         */
/*    Auto-fitter glyph loading routines (body).                           */
/*                                                                         */
/*  Copyright 2003-2015 by                                                 */
/*  David Turner, Robert Wilhelm, and Werner Lemberg.                      */
/*                                                                         */
/*  This file is part of the FreeType project, and may only be used,       */
/*  modified, and distributed under the terms of the FreeType project      */
/*  license, LICENSE.TXT.  By continuing to use, modify, or distribute     */
/*  this file you indicate that you have read the license and              */
/*  understand and accept it fully.                                        */
/*                                                                         */
/***************************************************************************/


#include "afglobal.h"
#include "afloader.h"
#include "afhints.h"
#include "aferrors.h"
#include "afmodule.h"
#include "afpic.h"


  /* Initialize glyph loader. */

  FT_LOCAL_DEF( void )
  af_loader_init( AF_Loader      loader,
                  AF_GlyphHints  hints )
  {
    FT_ZERO( loader );

    loader->hints = hints;
  }


  /* Reset glyph loader and compute globals if necessary. */

  FT_LOCAL_DEF( FT_Error )
  af_loader_reset( AF_Loader  loader,
                   AF_Module  module,
                   FT_Face    face )
  {
    FT_Error  error = FT_Err_Ok;


    loader->face    = face;
    loader->globals = (AF_FaceGlobals)face->autohint.data;

    if ( loader->globals == NULL )
    {
      error = af_face_globals_new( face, &loader->globals, module );
      if ( !error )
      {
        face->autohint.data =
          (FT_Pointer)loader->globals;
        face->autohint.finalizer =
          (FT_Generic_Finalizer)af_face_globals_free;
      }
    }

    return error;
  }


  /* Finalize glyph loader. */

  FT_LOCAL_DEF( void )
  af_loader_done( AF_Loader  loader )
  {
    loader->face    = NULL;
    loader->globals = NULL;
    loader->hints   = NULL;
  }


  /* Do the main work of `af_loader_load_glyph'.  Note that we never   */
  /* have to deal with composite glyphs as those get loaded into       */
  /* FT_GLYPH_FORMAT_OUTLINE by the recursed `FT_Load_Glyph' function. */
  /* In the rare cases where FT_LOAD_NO_RECURSE is set, it implies     */
  /* FT_LOAD_NO_SCALE and as such the auto-hinter is never called.     */

  static FT_Error
  af_loader_load_g( AF_Loader  loader,
                    AF_Scaler  scaler,
                    FT_UInt    glyph_index,
                    FT_Int32   load_flags )
  {
    FT_Error          error;
    FT_Face           face     = loader->face;
    AF_StyleMetrics   metrics  = loader->metrics;
    AF_GlyphHints     hints    = loader->hints;
    FT_GlyphSlot      slot     = face->glyph;
    FT_Slot_Internal  internal = slot->internal;
    FT_GlyphLoader    gloader  = internal->loader;
    FT_Int32          flags;


    flags = load_flags | FT_LOAD_LINEAR_DESIGN;
    error = FT_Load_Glyph( face, glyph_index, flags );
    if ( error )
      goto Exit;

    loader->transformed = internal->glyph_transformed;
    if ( loader->transformed )
    {
      FT_Matrix  inverse;


      loader->trans_matrix = internal->glyph_matrix;
      loader->trans_delta  = internal->glyph_delta;

      inverse = loader->trans_matrix;
      if ( !FT_Matrix_Invert( &inverse ) )
        FT_Vector_Transform( &loader->trans_delta, &inverse );
    }

    switch ( slot->format )
    {
    case FT_GLYPH_FORMAT_OUTLINE:
      /* translate the loaded glyph when an internal transform is needed */
      if ( loader->transformed )
        FT_Outline_Translate( &slot->outline,
                              loader->trans_delta.x,
                              loader->trans_delta.y );

      /* compute original horizontal phantom points (and ignore */
      /* vertical ones)                                         */
      loader->pp1.x = hints->x_delta;
      loader->pp1.y = hints->y_delta;
      loader->pp2.x = FT_MulFix( slot->metrics.horiAdvance,
                                 hints->x_scale ) + hints->x_delta;
      loader->pp2.y = hints->y_delta;

      /* be sure to check for spacing glyphs */
      if ( slot->outline.n_points == 0 )
        goto Hint_Metrics;

      /* now load the slot image into the auto-outline and run the */
      /* automatic hinting process                                 */
      {
#ifdef FT_CONFIG_OPTION_PIC
        AF_FaceGlobals         globals = loader->globals;
#endif
        AF_StyleClass          style_class = metrics->style_class;
        AF_WritingSystemClass  writing_system_class =
          AF_WRITING_SYSTEM_CLASSES_GET[style_class->writing_system];


        if ( writing_system_class->style_hints_apply )
          writing_system_class->style_hints_apply( glyph_index,
                                                   hints,
                                                   &gloader->base.outline,
                                                   metrics );
      }

      /* we now need to adjust the metrics according to the change in */
      /* width/positioning that occurred during the hinting process   */
      if ( scaler->render_mode != FT_RENDER_MODE_LIGHT )
      {
        FT_Pos        old_rsb, old_lsb, new_lsb;
        FT_Pos        pp1x_uh, pp2x_uh;
        AF_AxisHints  axis  = &hints->axis[AF_DIMENSION_HORZ];
        AF_Edge       edge1 = axis->edges;         /* leftmost edge  */
        AF_Edge       edge2 = edge1 +
                              axis->num_edges - 1; /* rightmost edge */


        if ( axis->num_edges > 1 && AF_HINTS_DO_ADVANCE( hints ) )
        {
          old_rsb = loader->pp2.x - edge2->opos;
          old_lsb = edge1->opos;
          new_lsb = edge1->pos;

          /* remember unhinted values to later account */
          /* for rounding errors                       */

          pp1x_uh = new_lsb    - old_lsb;
          pp2x_uh = edge2->pos + old_rsb;

          /* prefer too much space over too little space */
          /* for very small sizes                        */

          if ( old_lsb < 24 )
            pp1x_uh -= 8;

          if ( old_rsb < 24 )
            pp2x_uh += 8;

          loader->pp1.x = FT_PIX_ROUND( pp1x_uh );
          loader->pp2.x = FT_PIX_ROUND( pp2x_uh );

          if ( loader->pp1.x >= new_lsb && old_lsb > 0 )
            loader->pp1.x -= 64;

          if ( loader->pp2.x <= edge2->pos && old_rsb > 0 )
            loader->pp2.x += 64;

          slot->lsb_delta = loader->pp1.x - pp1x_uh;
          slot->rsb_delta = loader->pp2.x - pp2x_uh;
        }
        else
        {
          FT_Pos  pp1x = loader->pp1.x;
          FT_Pos  pp2x = loader->pp2.x;


          loader->pp1.x = FT_PIX_ROUND( pp1x );
          loader->pp2.x = FT_PIX_ROUND( pp2x );

          slot->lsb_delta = loader->pp1.x - pp1x;
          slot->rsb_delta = loader->pp2.x - pp2x;
        }
      }
      else
      {
        FT_Pos  pp1x = loader->pp1.x;
        FT_Pos  pp2x = loader->pp2.x;


        loader->pp1.x = FT_PIX_ROUND( pp1x + hints->xmin_delta );
        loader->pp2.x = FT_PIX_ROUND( pp2x + hints->xmax_delta );

        slot->lsb_delta = loader->pp1.x - pp1x;
        slot->rsb_delta = loader->pp2.x - pp2x;
      }

      break;

    default:
      /* we don't support other formats (yet?) */
      error = FT_THROW( Unimplemented_Feature );
    }

  Hint_Metrics:
    {
      FT_BBox    bbox;
      FT_Vector  vvector;


      vvector.x = slot->metrics.vertBearingX - slot->metrics.horiBearingX;
      vvector.y = slot->metrics.vertBearingY - slot->metrics.horiBearingY;
      vvector.x = FT_MulFix( vvector.x, metrics->scaler.x_scale );
      vvector.y = FT_MulFix( vvector.y, metrics->scaler.y_scale );

      /* transform the hinted outline if needed */
      if ( loader->transformed )
      {
        FT_Outline_Transform( &gloader->base.outline, &loader->trans_matrix );
        FT_Vector_Transform( &vvector, &loader->trans_matrix );
      }
#if 1
      /* we must translate our final outline by -pp1.x and compute */
      /* the new metrics                                           */
      if ( loader->pp1.x )
        FT_Outline_Translate( &gloader->base.outline, -loader->pp1.x, 0 );
#endif
      FT_Outline_Get_CBox( &gloader->base.outline, &bbox );

      bbox.xMin = FT_PIX_FLOOR( bbox.xMin );
      bbox.yMin = FT_PIX_FLOOR( bbox.yMin );
      bbox.xMax = FT_PIX_CEIL(  bbox.xMax );
      bbox.yMax = FT_PIX_CEIL(  bbox.yMax );

      slot->metrics.width        = bbox.xMax - bbox.xMin;
      slot->metrics.height       = bbox.yMax - bbox.yMin;
      slot->metrics.horiBearingX = bbox.xMin;
      slot->metrics.horiBearingY = bbox.yMax;

      slot->metrics.vertBearingX = FT_PIX_FLOOR( bbox.xMin + vvector.x );
      slot->metrics.vertBearingY = FT_PIX_FLOOR( bbox.yMax + vvector.y );

      /* for mono-width fonts (like Andale, Courier, etc.) we need */
      /* to keep the original rounded advance width; ditto for     */
      /* digits if all have the same advance width                 */
#if 0
      if ( !FT_IS_FIXED_WIDTH( slot->face ) )
        slot->metrics.horiAdvance = loader->pp2.x - loader->pp1.x;
      else
        slot->metrics.horiAdvance = FT_MulFix( slot->metrics.horiAdvance,
                                               x_scale );
#else
      if ( scaler->render_mode != FT_RENDER_MODE_LIGHT                      &&
           ( FT_IS_FIXED_WIDTH( slot->face )                              ||
             ( af_face_globals_is_digit( loader->globals, glyph_index ) &&
               metrics->digits_have_same_width                          ) ) )
      {
        slot->metrics.horiAdvance = FT_MulFix( slot->metrics.horiAdvance,
                                               metrics->scaler.x_scale );

        /* Set delta values to 0.  Otherwise code that uses them is */
        /* going to ruin the fixed advance width.                   */
        slot->lsb_delta = 0;
        slot->rsb_delta = 0;
      }
      else
      {
        /* non-spacing glyphs must stay as-is */
        if ( slot->metrics.horiAdvance )
          slot->metrics.horiAdvance = loader->pp2.x - loader->pp1.x;
      }
#endif

      slot->metrics.vertAdvance = FT_MulFix( slot->metrics.vertAdvance,
                                             metrics->scaler.y_scale );

      slot->metrics.horiAdvance = FT_PIX_ROUND( slot->metrics.horiAdvance );
      slot->metrics.vertAdvance = FT_PIX_ROUND( slot->metrics.vertAdvance );

#if 0
      /* reassign all outline fields except flags to protect them */
      slot->outline.n_contours = internal->loader->base.outline.n_contours;
      slot->outline.n_points   = internal->loader->base.outline.n_points;
      slot->outline.points     = internal->loader->base.outline.points;
      slot->outline.tags       = internal->loader->base.outline.tags;
      slot->outline.contours   = internal->loader->base.outline.contours;
#endif

      slot->format  = FT_GLYPH_FORMAT_OUTLINE;
    }

  Exit:
    return error;
  }


  /* Load a glyph. */

  FT_LOCAL_DEF( FT_Error )
  af_loader_load_glyph( AF_Loader  loader,
                        AF_Module  module,
                        FT_Face    face,
                        FT_UInt    gindex,
                        FT_Int32   load_flags )
  {
    FT_Error      error;
    FT_Size       size   = face->size;
    AF_ScalerRec  scaler;


    if ( !size )
      return FT_THROW( Invalid_Size_Handle );

    FT_ZERO( &scaler );

    scaler.face    = face;
    scaler.x_scale = size->metrics.x_scale;
    scaler.x_delta = 0;  /* XXX: TODO: add support for sub-pixel hinting */
    scaler.y_scale = size->metrics.y_scale;
    scaler.y_delta = 0;  /* XXX: TODO: add support for sub-pixel hinting */

    scaler.render_mode = FT_LOAD_TARGET_MODE( load_flags );
    scaler.flags       = 0;  /* XXX: fix this */

    error = af_loader_reset( loader, module, face );
    if ( !error )
    {
      AF_StyleMetrics  metrics;
      FT_UInt          options = AF_STYLE_NONE_DFLT;


#ifdef FT_OPTION_AUTOFIT2
      /* XXX: undocumented hook to activate the latin2 writing system */
      if ( load_flags & ( 1UL << 20 ) )
        options = AF_STYLE_LTN2_DFLT;
#endif

      error = af_face_globals_get_metrics( loader->globals, gindex,
                                           options, &metrics );
      if ( !error )
      {
#ifdef FT_CONFIG_OPTION_PIC
        AF_FaceGlobals         globals = loader->globals;
#endif
        AF_StyleClass          style_class = metrics->style_class;
        AF_WritingSystemClass  writing_system_class =
          AF_WRITING_SYSTEM_CLASSES_GET[style_class->writing_system];


        loader->metrics = metrics;

        if ( writing_system_class->style_metrics_scale )
          writing_system_class->style_metrics_scale( metrics, &scaler );
        else
          metrics->scaler = scaler;

        load_flags |=  FT_LOAD_NO_SCALE | FT_LOAD_IGNORE_TRANSFORM;
        load_flags &= ~FT_LOAD_RENDER;

        if ( writing_system_class->style_hints_init )
        {
          error = writing_system_class->style_hints_init( loader->hints,
                                                          metrics );
          if ( error )
            goto Exit;
        }

        error = af_loader_load_g( loader, &scaler, gindex, load_flags );
      }
    }
  Exit:
    return error;
  }


/* END */
